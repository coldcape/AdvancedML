"""
deep_FNN_cifar10_sharding.py

This script trains a deep feedforward neural network (MLP) on the CIFAR-10 dataset
using JAX with multi-GPU data parallelism via the pjit API.

Key points:
- Environment variables and JAX configuration are set at the very beginning to force GPU usage.
- A 1D device mesh is created using the first 2 GPUs.
- The model parameters and optimizer state are replicated across devices.
- Input mini-batches are sharded along the batch dimension so that each GPU processes half of the batch.
- The number of mini-batches per epoch is capped at 100.
- After each epoch, the model is evaluated on the test set, and both test loss and test accuracy are printed.
- Detailed comments explain each major code section and the reasons behind the choices.
"""

# --------------------------------------------------------------------------
# Set environment variables before any JAX imports to force GPU usage.
# --------------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"           # Make GPUs 0 and 1 visible.
os.environ["JAX_PLATFORM_NAME"] = "gpu"              # Force JAX to use the GPU platform.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Do not preallocate all GPU memory.

# Import required libraries.
import pickle
import time  # For timing measurements
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import optax

# ----------- Additional imports for sharding via pjit -----------
# pjit compiles and executes computations in parallel over devices.
from jax.experimental.pjit import pjit
# Mesh and PartitionSpec are used to specify how arrays are mapped to devices.
from jax.sharding import Mesh, PartitionSpec, NamedSharding
# ------------------------------------------------------------------

# ------------------------- Data Loading -------------------------
def unpickle(file):
    """
    Unpickle a given file and return the loaded dictionary.
    
    Args:
        file (str): Path to the pickle file.
    
    Returns:
        dict: The unpickled dictionary.
    """
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10(data_dir="cifar-10-batches-py"):
    """
    Load the CIFAR-10 dataset from a specified directory.
    
    The function:
      1. Loads the 5 training batches and concatenates them.
      2. Loads the test batch.
      3. Reshapes and transposes the data to shape (N, 32, 32, 3).
      4. Normalizes pixel values to the range [0, 1].
    
    Args:
        data_dir (str): Directory containing CIFAR-10 batch files.
    
    Returns:
        Tuple of numpy arrays: (train_data, train_labels, test_data, test_labels)
    """
    train_data_list, train_labels_list = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        train_data_list.append(batch[b"data"])
        train_labels_list.extend(batch[b"labels"])
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.array(train_labels_list)
    # Reshape from (N, 3072) to (N, 3, 32, 32) and then transpose to (N, 32, 32, 3)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_data = train_data.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    test_data = test_batch[b"data"]
    test_labels = np.array(test_batch[b"labels"])
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.astype(np.float32) / 255.0
    
    print("CIFAR-10 dataset loaded.")
    print(" Training data shape:", train_data.shape)
    print(" Test data shape:", test_data.shape)
    return train_data, train_labels, test_data, test_labels

# ------------------------- Model Initialization -------------------------
def initialize_params(key, layer_sizes):
    """
    Initialize the network parameters (weights, biases, and batch norm parameters).
    
    Uses He (Kaiming) initialization for the weights.
    
    Args:
        key (jax.random.PRNGKey): JAX PRNG key for random number generation.
        layer_sizes (list): List of layer sizes (e.g., [3072, 5000, 4000, 3000, 2000, 1000, 10]).
    
    Returns:
        list: A list of tuples, each tuple containing (W, b, gamma, beta) for a layer.
    """
    print("[DEBUG] Initializing model parameters...")
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        # Weight initialization: random normal scaled by sqrt(2 / fan_in)
        W = random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        # Batch norm parameters: initialize gamma=1 and beta=0
        gamma = jnp.ones((layer_sizes[i+1],))
        beta = jnp.zeros((layer_sizes[i+1],))
        params.append((W, b, gamma, beta))
    return params

# ------------------------- Model Forward Pass -------------------------
def batch_norm(x, gamma, beta):
    """
    Apply batch normalization to input x.
    
    Args:
        x (jnp.ndarray): Input array.
        gamma (jnp.ndarray): Scale parameter.
        beta (jnp.ndarray): Shift parameter.
    
    Returns:
        jnp.ndarray: The normalized, scaled, and shifted output.
    """
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * x_norm + beta

def forward(params, x):
    """
    Compute the forward pass of the MLP.
    
    For each hidden layer, a linear transformation is applied, followed by batch normalization
    and the ReLU activation. The final layer applies only a linear transformation.
    
    Args:
        params (list): List of model parameters.
        x (jnp.ndarray): Input data (flattened images).
    
    Returns:
        jnp.ndarray: The logits from the final layer.
    """
    for (W, b, gamma, beta) in params[:-1]:
        x = jnp.dot(x, W) + b           # Linear transformation
        x = batch_norm(x, gamma, beta)    # Batch normalization
        x = jax.nn.relu(x)              # ReLU activation
    W_out, b_out, _, _ = params[-1]
    logits = jnp.dot(x, W_out) + b_out   # Final linear layer
    return logits

# ------------------------- Loss and Accuracy -------------------------
def cross_entropy_loss(params, x, y, l2_lambda=5e-5):
    """
    Compute the cross-entropy loss with L2 regularization.
    
    Args:
        params (list): Model parameters.
        x (jnp.ndarray): Input data.
        y (jnp.ndarray): True labels.
        l2_lambda (float): L2 regularization coefficient.
    
    Returns:
        float: The total loss.
    """
    logits = forward(params, x)
    y_one_hot = jax.nn.one_hot(y, num_classes=10)
    ce_loss = -jnp.mean(jnp.sum(y_one_hot * jax.nn.log_softmax(logits), axis=-1))
    l2_loss = sum(jnp.sum(W**2) for W, _, _, _ in params) * l2_lambda
    return ce_loss + l2_loss

def compute_accuracy(params, x, y):
    """
    Compute the accuracy of the model.
    
    Args:
        params (list): Model parameters.
        x (jnp.ndarray): Input data.
        y (jnp.ndarray): True labels.
    
    Returns:
        float: The fraction of correct predictions.
    """
    logits = forward(params, x)
    predictions = jnp.argmax(jax.nn.softmax(logits), axis=-1)
    return jnp.mean(predictions == y)

# ------------------------- Optimizer Setup -------------------------
def create_optimizer(base_lr=0.001, decay_rate=0.98, decay_steps=100):
    """
    Create an Adam optimizer with an exponential learning rate decay schedule.
    
    Args:
        base_lr (float): Initial learning rate.
        decay_rate (float): Decay factor.
        decay_steps (int): Number of steps between decays.
    
    Returns:
        optax.GradientTransformation: The configured optimizer.
    """
    schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    return optax.adam(schedule)

# Global optimizer instance.
optimizer = create_optimizer()

# ------------------------- Sharded Training Step with pjit -------------------------
def train_mlp_sharded(train_data, train_labels, test_data, test_labels, num_epochs=200, batch_size=128):
    """
    Train the network using multi-GPU data parallelism via pjit.
    
    This function:
      1. Sets up a device mesh using the first 2 GPUs.
      2. Replicates the model parameters and optimizer state across devices.
      3. Shards the input mini-batches along the batch dimension so that each GPU processes half the batch.
      4. Caps the number of mini-batches per epoch to at most 100.
      5. After each epoch, evaluates the model on the test set, printing test loss, test accuracy, and epoch runtime.
    
    Args:
        train_data, train_labels, test_data, test_labels: CIFAR-10 data.
        num_epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
    
    Returns:
        None
    """
    print("\n[DEBUG] Entering sharded training function (train_mlp_sharded)...")
    train_loop_start = time.time()

    # Create a device mesh using the first 2 GPUs.
    all_devices = jax.devices()
    mesh_devices = all_devices[:2]  # Use the first 2 GPUs.
    mesh = Mesh(mesh_devices, ('devices',))  # 'devices' is the logical axis name.

    num_train = train_data.shape[0]
    # Cap the number of mini-batches per epoch at 100.
    num_batches = min(num_train // batch_size, 100)

    # Initialize model parameters.
    key = random.PRNGKey(42)
    layer_sizes = [3072, 5000, 4000, 3000, 2000, 1000, 10]
    params = initialize_params(key, layer_sizes)
    opt_state = optimizer.init(params)

    # Define sharding specifications:
    # - param_sharding: replicate parameters on both GPUs.
    # - data_sharding: shard the batch dimension across GPUs.
    global param_sharding, opt_state_sharding, data_sharding
    param_sharding = NamedSharding(mesh, PartitionSpec())          # Replicated across devices.
    opt_state_sharding = NamedSharding(mesh, PartitionSpec())       # Replicated.
    data_sharding = NamedSharding(mesh, PartitionSpec('devices'))   # Shard the first dimension (batch).

    # Define the update step function.
    def step_fn(params, opt_state, x, y):
        loss_value, grads = value_and_grad(cross_entropy_loss)(params, x, y)
        # Clip gradients to stabilize training.
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # Compile the update step using pjit (only once).
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
        # Place the parameters and optimizer state onto the device mesh.
        params = jax.device_put(params, param_sharding)
        opt_state = jax.device_put(opt_state, opt_state_sharding)
        # Training loop over epochs.
        for epoch in range(num_epochs):
            epoch_start = time.time()
            # Shuffle the training data and labels.
            perm = np.random.permutation(num_train)
            train_data = train_data[perm]
            train_labels = train_labels[perm]
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                # Extract the current mini-batch and flatten the images.
                x_batch = train_data[start_idx:end_idx].reshape(batch_size, -1)
                y_batch = train_labels[start_idx:end_idx]
                # Shard the mini-batch across devices.
                x_shard = jax.device_put(x_batch, data_sharding)
                y_shard = jax.device_put(y_batch, data_sharding)
                # Execute the parallel update step.
                params, opt_state, loss_value = pjit_step(params, opt_state, x_shard, y_shard)
            epoch_time = time.time() - epoch_start
            # Evaluate on test data at the end of each epoch.
            test_loss = cross_entropy_loss(params, test_data.reshape(test_data.shape[0], -1), test_labels)
            test_acc = compute_accuracy(params, test_data.reshape(test_data.shape[0], -1), test_labels)
            print(f"Epoch {epoch+1:03d} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}% | Epoch Time: {epoch_time:.2f} s")
    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"[DEBUG] Total training (epochs) time (2-GPU sharded): {total_training_time:.2f} seconds")
    
    # Final evaluation on test set.
    final_test_loss = cross_entropy_loss(params, test_data.reshape(test_data.shape[0], -1), test_labels)
    final_test_acc = compute_accuracy(params, test_data.reshape(test_data.shape[0], -1), test_labels)
    print(f"\nFinal Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")

# ------------------------- Main Execution -------------------------
if __name__ == "__main__":
    script_start = time.time()

    # Print available devices (should show GPUs).
    print("\n[DEBUG] Available JAX devices at script start:", jax.devices())

    # Load CIFAR-10 dataset.
    train_data, train_labels, test_data, test_labels = load_cifar10()
    
    # Run the sharded (parallel, multi-GPU) training.
    # For demonstration purposes, we run for 100 epochs.
    train_mlp_sharded(train_data, train_labels, test_data, test_labels, num_epochs=100, batch_size=128)
    
    script_end = time.time()
    print(f"\n[DEBUG] Total script execution time: {script_end - script_start:.2f} seconds")
    print("[DEBUG] Script execution complete!")
