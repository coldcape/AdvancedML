# deep_FNN_cifar100_sharding.py

"""
deep_FNN_cifar100_sharding.py

This script trains a deep feedforward neural network (MLP) on the CIFAR-100 dataset
using JAX with multi-GPU data parallelism via the pjit API.

Key points:
- A 1D mesh is created using the first 2 GPUs.
- The model parameters and optimizer state are replicated.
- Each mini-batch is sharded along its leading dimension so that each GPU processes half the batch.
- The number of mini-batches per epoch is capped at 100 (for demonstration).
- After each epoch, test loss and accuracy are printed, along with epoch runtime.
- The final layer has 100 outputs for CIFAR-100.
- We load CIFAR-100 from single files 'train' and 'test' in 'cifar-100-batches-py'.
"""

import os
# Force JAX to use two GPUs (0,1) if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax

# pjit = parallel just-in-time transformation
from jax.experimental.pjit import pjit
# Mesh defines how we map logical axes to physical devices
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.tree_util

# ------------------------- Data Loading (CIFAR-100) ------------------------- #
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_cifar100(data_dir="cifar-100-batches-py"):
    """
    Load the CIFAR-100 dataset from single 'train' and 'test' files in `data_dir`.
    
    Each file is a pickled dictionary containing:
        - b'data': array of shape (N, 3072)
        - b'fine_labels': list of length N (values in [0..99])
    
    Steps:
    1. Load 'train' -> shape (50000, 3072)
    2. Load 'test' -> shape (10000, 3072)
    3. Reshape to (N, 32, 32, 3)
    4. Normalize pixel values to [0,1]
    """
    # Load training set
    train_dict = unpickle(os.path.join(data_dir, "train"))
    train_data = train_dict[b'data']
    train_labels = np.array(train_dict[b'fine_labels'])

    # Reshape from (N, 3072) to (N, 32, 32, 3)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_data = train_data.astype(np.float32) / 255.0

    # Load test set
    test_dict = unpickle(os.path.join(data_dir, "test"))
    test_data = test_dict[b'data']
    test_labels = np.array(test_dict[b'fine_labels'])

    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.astype(np.float32) / 255.0

    print("CIFAR-100 dataset loaded.")
    print(" Training data shape:", train_data.shape)
    print(" Test data shape:", test_data.shape)
    return train_data, train_labels, test_data, test_labels

# ------------------------- Model Initialization ------------------------- #
def initialize_params(key, layer_sizes):
    """
    Initialize (W, b, gamma, beta) for each layer using He initialization.
    """
    print("[DEBUG] Initializing model parameters...")
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        W = random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) \
            * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        gamma = jnp.ones((layer_sizes[i+1],))
        beta = jnp.zeros((layer_sizes[i+1],))
        params.append((W, b, gamma, beta))
    return params

# ------------------------- Model Forward Pass ------------------------- #
def batch_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * x_norm + beta

def forward(params, x):
    """
    For each hidden layer: x = BN( xW + b ), then ReLU.
    Final layer is just xW + b (logits).
    """
    for (W, b, gamma, beta) in params[:-1]:
        x = jnp.dot(x, W) + b
        x = batch_norm(x, gamma, beta)
        x = jax.nn.relu(x)
    W_out, b_out, _, _ = params[-1]
    logits = jnp.dot(x, W_out) + b_out
    return logits

# ------------------------- Loss and Accuracy (CIFAR-100) ------------------------- #
def cross_entropy_loss(params, x, y, l2_lambda=5e-5):
    """
    For CIFAR-100, we have num_classes=100. Uses one-hot encoding and log_softmax.
    """
    logits = forward(params, x)
    y_one_hot = jax.nn.one_hot(y, num_classes=100)
    ce_loss = -jnp.mean(jnp.sum(y_one_hot * jax.nn.log_softmax(logits), axis=-1))
    l2_loss = sum(jnp.sum(W**2) for W, _, _, _ in params) * l2_lambda
    return ce_loss + l2_loss

def compute_accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(jax.nn.softmax(logits), axis=-1)
    return jnp.mean(preds == y)

# ------------------------- Optimizer Setup ------------------------- #
def create_optimizer(base_lr=0.001, decay_rate=0.98, decay_steps=100):
    """
    Adam optimizer with exponential decay schedule.
    """
    schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    return optax.adam(schedule)

optimizer = create_optimizer()

# ------------------------- Sharded/Parallel Training with pjit ------------------------- #
def train_mlp_sharded(train_data, train_labels, test_data, test_labels,
                      num_epochs=100, batch_size=128):
    """
    Train a multi-GPU (2-GPU) CIFAR-100 MLP via pjit.
    
    Steps:
      1. Create a mesh of 2 devices.
      2. Replicate params & opt_state.
      3. Shard each mini-batch so each GPU sees half the data.
      4. Cap at 100 mini-batches per epoch (for demonstration).
      5. After each epoch, print test loss/acc/time.
      6. Final layer = 100 outputs for CIFAR-100.
    """
    print("\n[DEBUG] Entering sharded training function (train_mlp_sharded) for CIFAR-100...")
    train_loop_start = time.time()

    # Create a device mesh using the first 2 GPUs
    all_devices = jax.devices()
    mesh_devices = all_devices[:2]  # use first 2 GPUs only
    mesh = Mesh(mesh_devices, ('devices',))

    num_train = train_data.shape[0]
    num_batches = min(num_train // batch_size, 100)  # cap at 100 mini-batches
    key = random.PRNGKey(42)

    # CIFAR-100 network architecture (final layer = 100)
    layer_sizes = [32*32*3, 5000, 4000, 3000, 2000, 1000, 100]
    params = initialize_params(key, layer_sizes)
    opt_state = optimizer.init(params)

    # Define sharding specs
    #   - param_sharding/opt_state_sharding: replicate across devices
    #   - data_sharding: shard along the "devices" dimension
    global param_sharding, opt_state_sharding, data_sharding
    param_sharding = NamedSharding(mesh, PartitionSpec())
    opt_state_sharding = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec('devices'))

    def step_fn(params, opt_state, x, y):
        loss_value, grads = value_and_grad(cross_entropy_loss)(params, x, y)
        # Simple gradient clipping
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # pjit-compiled function
    pjit_step = pjit(
        step_fn,
        in_shardings=(param_sharding, opt_state_sharding, data_sharding, data_sharding),
        out_shardings=(param_sharding, opt_state_sharding, None)
    )

    training_start = time.time()
    setup_time = training_start - train_loop_start
    print(f"[DEBUG] Setup time (within train_mlp_sharded): {setup_time:.2f} seconds")
    print("\n-------- Starting Parallel (Sharded) Training Across Two GPUs --------")

    with mesh:
        # Replicate params & opt_state
        params = jax.device_put(params, param_sharding)
        opt_state = jax.device_put(opt_state, opt_state_sharding)

        for epoch in range(num_epochs):
            epoch_start = time.time()
            # Shuffle the training data each epoch
            perm = np.random.permutation(num_train)
            train_data = train_data[perm]
            train_labels = train_labels[perm]

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                x_batch = train_data[start_idx:end_idx].reshape(batch_size, -1)
                y_batch = train_labels[start_idx:end_idx]

                # Shard the batch
                x_shard = jax.device_put(x_batch, data_sharding)
                y_shard = jax.device_put(y_batch, data_sharding)

                params, opt_state, loss_value = pjit_step(params, opt_state, x_shard, y_shard)

            epoch_time = time.time() - epoch_start
            # Evaluate on test (logits)
            test_loss = cross_entropy_loss(params, test_data.reshape(test_data.shape[0], -1), test_labels)
            test_acc = compute_accuracy(params, test_data.reshape(test_data.shape[0], -1), test_labels)
            print(f"Epoch {epoch+1:03d} | Test Loss: {test_loss:.4f} | "
                  f"Test Accuracy: {test_acc*100:.2f}% | Epoch Time: {epoch_time:.2f} s")

    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"[DEBUG] Total training (epochs) time (2-GPU sharded): {total_training_time:.2f} seconds")

    # Final evaluation
    final_test_loss = cross_entropy_loss(params, test_data.reshape(test_data.shape[0], -1), test_labels)
    final_test_acc = compute_accuracy(params, test_data.reshape(test_data.shape[0], -1), test_labels)
    print(f"\nFinal Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")

# ------------------------- Main Execution ------------------------- #
if __name__ == "__main__":
    script_start = time.time()

    # Print available devices
    print("\n[DEBUG] Available JAX devices at script start:", jax.devices())

    # Load CIFAR-100
    train_data, train_labels, test_data, test_labels = load_cifar100()

    # Train for 100 epochs (sharded across 2 GPUs)
    train_mlp_sharded(
        train_data, train_labels,
        test_data, test_labels,
        num_epochs=100,
        batch_size=128
    )

    script_end = time.time()
    print(f"\n[DEBUG] Total script execution time: {script_end - script_start:.2f} seconds")
    print("[DEBUG] Script execution complete!")
