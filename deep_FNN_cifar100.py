#!/usr/bin/env python3

"""
deep_FNN_cifar100.py

This script trains a deep feedforward neural network (MLP) on the CIFAR-100 dataset
using JAX in a single-device, non-parallel manner. It simply jit-compiles the
training step but does NOT shard or pmap the model/data across multiple GPUs.

Key changes from the CIFAR-10 version:
- We load the CIFAR-100 dataset from the single 'train' and 'test' files.
- The final layer has 100 output neurons (one for each CIFAR-100 fine class).
- We use jax.nn.one_hot(..., num_classes=100).

Other details remain similar: large hidden layers, batch norm, He init, Adam optimizer, etc.
"""

import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax
import jax.tree_util

# ------------------------- Data Loading (CIFAR-100) ------------------------- #

def unpickle(file):
    """
    Unpickle a given file using Python's pickle.
    """
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar100(data_dir="cifar-100-batches-py"):
    """
    Load the CIFAR-100 dataset from the specified directory.

    CIFAR-100 has:
      - train file: 'train'
      - test file: 'test'
      - Each file has keys b'data' (50000 for train, 10000 for test),
                    b'fine_labels' (100-class labels), etc.

    We do:
      1. Load 'train' -> 50000 images, shape (50000, 3072).
      2. Load 'test' -> 10000 images, shape (10000, 3072).
      3. Reshape to (N, 32, 32, 3).
      4. Normalize pixel values to [0,1].

    Returns:
      train_data, train_labels, test_data, test_labels
    """
    # Load training set
    train_dict = unpickle(os.path.join(data_dir, "train"))
    train_data = train_dict[b'data']
    train_labels = np.array(train_dict[b'fine_labels'])

    # Reshape from (50000, 3072) to (50000, 32, 32, 3)
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
    Initialize MLP parameters (W, b, gamma, beta) using He (Kaiming) init.
    Example layer_sizes: [3072, 5000, 4000, 3000, 2000, 1000, 100].
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
    For each layer (except last), apply linear->BN->ReLU.
    Last layer is linear only.
    """
    for (W, b, gamma, beta) in params[:-1]:
        x = jnp.dot(x, W) + b
        x = batch_norm(x, gamma, beta)
        x = jax.nn.relu(x)
    W_out, b_out, _, _ = params[-1]
    logits = jnp.dot(x, W_out) + b_out
    return logits

# ------------------------- Loss & Accuracy ------------------------- #

def cross_entropy_loss(params, x, y, l2_lambda=5e-5):
    """
    Cross-entropy loss with L2 regularization. For CIFAR-100, we do num_classes=100.
    """
    logits = forward(params, x)
    y_one_hot = jax.nn.one_hot(y, num_classes=100)
    ce_loss = -jnp.mean(jnp.sum(y_one_hot * jax.nn.log_softmax(logits), axis=-1))

    # L2 penalty
    l2_loss = sum(jnp.sum(W**2) for W, _, _, _ in params) * l2_lambda
    return ce_loss + l2_loss

def compute_accuracy(params, x, y):
    """
    Compute classification accuracy on CIFAR-100 (100 classes).
    """
    logits = forward(params, x)
    preds = jnp.argmax(jax.nn.softmax(logits), axis=-1)
    return jnp.mean(preds == y)

# ------------------------- Optimizer Setup ------------------------- #

def create_optimizer(base_lr=0.001, decay_rate=0.98, decay_steps=100):
    """
    Returns an Adam optimizer with exponential decay of the learning rate.
    """
    schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    return optax.adam(schedule)

optimizer = create_optimizer()

# ------------------------- Single-Device Training Step ------------------------- #

@jit
def update(params, opt_state, x, y):
    """
    Single-device update step for CIFAR-100 classification.
      1) compute cross-entropy loss & grads
      2) clip grads
      3) apply Adam updates
    """
    loss_val, grads = value_and_grad(cross_entropy_loss)(params, x, y)
    # Example: gradient clipping by value, just as a precaution
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

# ------------------------- Training Loop ------------------------- #

def train_mlp(train_data, train_labels, test_data, test_labels,
              num_epochs=100, batch_size=128):
    """
    Train MLP on single device for CIFAR-100 dataset.
    """
    train_loop_start = time.time()
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    key = random.PRNGKey(42)

    # MLP architecture for CIFAR-100 (100 classes)
    layer_sizes = [32*32*3, 5000, 4000, 3000, 2000, 1000, 100]
    params = initialize_params(key, layer_sizes)
    opt_state = optimizer.init(params)

    training_start = time.time()
    setup_time = training_start - train_loop_start
    print(f"Setup time (within train_mlp): {setup_time:.2f} seconds")

    print("\nStarting single-device training on CIFAR-100...")
    for epoch in range(num_epochs):
        # Shuffle each epoch
        perm = np.random.permutation(num_train)
        train_data = train_data[perm]
        train_labels = train_labels[perm]

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x_batch = train_data[start_idx:end_idx].reshape(batch_size, -1)
            y_batch = train_labels[start_idx:end_idx]
            params, opt_state, _ = update(params, opt_state, x_batch, y_batch)

        # Evaluate after epoch
        test_loss = cross_entropy_loss(
            params, test_data.reshape(test_data.shape[0], -1), test_labels
        )
        test_acc = compute_accuracy(
            params, test_data.reshape(test_data.shape[0], -1), test_labels
        )
        print(f"Epoch {epoch+1:02d} | Test Loss: {test_loss:.4f} | "
              f"Test Accuracy: {test_acc*100:.2f}%")

    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"Total training (epochs) time: {total_training_time:.2f} seconds")

# ------------------------- Main Execution ------------------------- #

if __name__ == "__main__":
    script_start = time.time()

    # Load CIFAR-100 data from 'cifar-100-batches-py'
    train_data, train_labels, test_data, test_labels = load_cifar100(
        data_dir="cifar-100-batches-py"
    )
    data_loading_end = time.time()

    print("\nTraining on CIFAR-100 (single device, no parallelism)...")
    train_mlp_start = time.time()

    # Train for 100 epochs with batch_size=128
    train_mlp(train_data, train_labels, test_data, test_labels,
              num_epochs=100, batch_size=128)

    train_mlp_end = time.time()

    print(f"\nTime to load data: {data_loading_end - script_start:.2f} seconds")
    print(f"Time in train_mlp function: {train_mlp_end - train_mlp_start:.2f} seconds")

    script_end = time.time()
    print(f"\nTotal script execution time: {script_end - script_start:.2f} seconds")
    print("[DEBUG] Script execution complete!")
