"""
deep_FNN_cifar10.py

This script trains a deep feedforward neural network (MLP) on the CIFAR-10 dataset
using JAX in a single-device, non-parallel manner. It simply jit-compiles the
training step but does NOT shard or pmap the model/data across multiple GPUs.

Key points:
- No references to pmap, pjit, or multi-GPU distribution.
- The code uses a single "update" step that is decorated with @jit.
- The number of mini-batches per epoch is not explicitly capped here; it uses the full dataset.
- We removed references to "module load cuda" or any HPC7 node requirement.
"""

import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax


# ------------------------- Data Loading ------------------------- #
def unpickle(file):
    """
    Unpickle the given file using Python's pickle.
    Loads serialized Python objects from a binary file.

    Args:
        file (str): File path to the pickle file.
    Returns:
        data_dict (dict): A dictionary containing the unpickled data.
    """
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10(data_dir="cifar-10-batches-py"):
    """
    Load the CIFAR-10 dataset from the provided directory, which contains
    several data batches in a Python pickle format. CIFAR-10 has 60,000 images:
    50,000 for training and 10,000 for testing. Each image is 32x32 pixels in RGB.

    1. Read training batches (data_batch_1 to data_batch_5).
    2. Concatenate them into a single training set (train_data, train_labels).
    3. Read the test batch (test_batch).
    4. Reshape data into (N, 32, 32, 3) format, where N is the number of images.
    5. Normalize pixel values to [0,1] by dividing by 255.0.

    Args:
        data_dir (str): Directory where CIFAR-10 batch files are located.

    Returns:
        (train_data, train_labels, test_data, test_labels)
    """
    train_data_list, train_labels_list = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        train_data_list.append(batch[b"data"])
        train_labels_list.extend(batch[b"labels"])

    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.array(train_labels_list)

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_data = train_data.astype(np.float32) / 255.0

    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    test_data = test_batch[b"data"]
    test_labels = np.array(test_batch[b"labels"])
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.astype(np.float32) / 255.0

    print("CIFAR-10 dataset loaded.")
    print(" Training data shape:", train_data.shape)
    print(" Test data shape:", test_data.shape)
    return train_data, train_labels, test_data, test_labels


# ------------------------- Model Initialization ------------------------- #
def initialize_params(key, layer_sizes):
    """
    Initialize weights, biases, and batch norm parameters (gamma and beta)
    for a multi-layer perceptron (MLP).

    Uses He (Kaiming) initialization for the weights. Each layer returns (W, b, gamma, beta).
    The final layer also has gamma, beta but they might not be used if we skip BN there.

    Args:
        key (jax.random.PRNGKey): Random number generator key for reproducibility.
        layer_sizes (list): Sizes of each layer, e.g., [3072, 5000, 4000, ... , 10].
    Returns:
        params (list): A list of (W, b, gamma, beta) for each layer.
    """
    params = []
    keys = random.split(key, len(layer_sizes) - 1)

    for i in range(len(layer_sizes) - 1):
        # Weight matrix
        W = random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        gamma = jnp.ones((layer_sizes[i+1],))
        beta = jnp.zeros((layer_sizes[i+1],))
        params.append((W, b, gamma, beta))

    return params

# ------------------------- Model Forward ------------------------- #
def batch_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * x_norm + beta

def forward(params, x):
    for (W, b, gamma, beta) in params[:-1]:
        x = jnp.dot(x, W) + b
        x = batch_norm(x, gamma, beta)
        x = jax.nn.relu(x)
    W_out, b_out, _, _ = params[-1]
    logits = jnp.dot(x, W_out) + b_out
    return logits

# ------------------------- Loss & Accuracy ------------------------- #
def cross_entropy_loss(params, x, y, l2_lambda=5e-5):
    logits = forward(params, x)
    y_one_hot = jax.nn.one_hot(y, num_classes=10)
    ce_loss = -jnp.mean(jnp.sum(y_one_hot * jax.nn.log_softmax(logits), axis=-1))
    l2_loss = sum(jnp.sum(W**2) for W, _, _, _ in params) * l2_lambda
    return ce_loss + l2_loss

def compute_accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(jax.nn.softmax(logits), axis=-1)
    return jnp.mean(preds == y)

# ------------------------- Optimizer Setup ------------------------- #
def create_optimizer(base_lr=0.001, decay_rate=0.98, decay_steps=100):
    schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    return optax.adam(schedule)

optimizer = create_optimizer()

# ------------------------- Training Step ------------------------- #
@jit
def update(params, opt_state, x, y):
    """
    Single-device update step. JIT-compiled.
      1) compute loss & grads
      2) clip grads
      3) apply Adam updates
    """
    loss_val, grads = value_and_grad(cross_entropy_loss)(params, x, y)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

# ------------------------- Training Loop ------------------------- #
def train_mlp(train_data, train_labels, test_data, test_labels, num_epochs=100, batch_size=128):
    """
    Train MLP on single device (CPU or GPU) for performance comparison.
    """
    train_loop_start = time.time()

    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    key = random.PRNGKey(42)

    # Define network
    layer_sizes = [3072, 5000, 4000, 3000, 2000, 1000, 10]
    params = initialize_params(key, layer_sizes)
    opt_state = optimizer.init(params)

    training_start = time.time()
    setup_time = training_start - train_loop_start
    print(f"Setup time (within train_mlp): {setup_time:.2f} seconds")

    print("\nStarting training (single-device)...")
    for epoch in range(num_epochs):
        # Shuffle data each epoch
        perm = np.random.permutation(num_train)
        train_data, train_labels = train_data[perm], train_labels[perm]

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x_batch = train_data[start_idx:end_idx].reshape(batch_size, -1)
            y_batch = train_labels[start_idx:end_idx]

            params, opt_state, _ = update(params, opt_state, x_batch, y_batch)

        # Evaluate on test set each epoch
        test_loss = cross_entropy_loss(params, test_data.reshape(test_data.shape[0], -1), test_labels)
        test_acc = compute_accuracy(params, test_data.reshape(test_data.shape[0], -1), test_labels)
        print(f"Epoch {epoch+1:02d} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"Total training (epochs) time: {total_training_time:.2f} seconds")


# ------------------------- Main Execution ------------------------- #
if __name__ == "__main__":
    script_start = time.time()

    # Load CIFAR-10 data
    train_data, train_labels, test_data, test_labels = load_cifar10()
    data_loading_end = time.time()

    print("\nTraining on a single device (CPU or one GPU, no parallelism)...")

    train_mlp_start = time.time()
    # Train
    train_mlp(train_data, train_labels, test_data, test_labels, num_epochs=100, batch_size=128)
    train_mlp_end = time.time()

    print(f"\nTime to load data (script start -> data load end): {data_loading_end - script_start:.2f} seconds")
    print(f"Time spent in train_mlp function: {train_mlp_end - train_mlp_start:.2f} seconds")

    script_end = time.time()
    print(f"\nTotal script execution time: {script_end - script_start:.2f} seconds")
