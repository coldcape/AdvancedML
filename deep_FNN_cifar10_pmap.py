"""
deep_FNN_cifar10_pmap.py

This script trains a deep feedforward neural network (MLP) on the CIFAR-10 dataset
using JAX with multi-GPU data parallelism via pmap.

Key points:
- We replicate parameters & optimizer state across all devices.
- We use pmap to parallelize the update function, automatically splitting batches across devices.
- The total batch size should be divisible by the number of GPUs (e.g., 2).
- The number of mini-batches per epoch is capped at 100.
- We fix the "not enough values to unpack" error by extracting parameters
  from device 0 with `jax.tree_map(lambda x: x[0], params)` at test time.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"             # Make GPUs 0 and 1 visible
os.environ["JAX_PLATFORM_NAME"] = "gpu"                # Force JAX to use GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable full GPU memory preallocation

import pickle
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax

from functools import partial

# ------------------------- Data Loading -------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_cifar10(data_dir="cifar-10-batches-py"):
    train_data_list, train_labels_list = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        train_data_list.append(batch[b"data"])
        train_labels_list.extend(batch[b"labels"])

    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.array(train_labels_list)
    # Reshape to (N, 32, 32, 3)
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

# ------------------------- Model Initialization -------------------------
def initialize_params(key, layer_sizes):
    """
    Initialize (W, b, gamma, beta) for each layer using He (Kaiming) initialization.
    """
    print("[DEBUG] Initializing model parameters...")
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        W = random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        gamma = jnp.ones((layer_sizes[i+1],))
        beta = jnp.zeros((layer_sizes[i+1],))
        params.append((W, b, gamma, beta))
    return params

# ------------------------- Forward Pass -------------------------
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
    return jnp.dot(x, W_out) + b_out

# ------------------------- Loss & Accuracy -------------------------
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

# ------------------------- Optimizer Setup -------------------------
def create_optimizer(base_lr=0.001, decay_rate=0.98, decay_steps=100):
    schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    return optax.adam(schedule)

optimizer = create_optimizer()

# ------------------------- pmap-based Update -------------------------
def loss_fn(params, x, y):
    return cross_entropy_loss(params, x, y)

@partial(jax.pmap, axis_name='batch')
def update_fn(params, opt_state, x, y):
    """
    pmap update:
      1) compute loss & grads
      2) average grads across devices with pmean
      3) apply updates via Adam
    """
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    grads = jax.lax.pmean(grads, axis_name='batch')  # average gradients across devices
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ------------------------- Main Training Loop (pmap) -------------------------
def train_mlp_pmap(train_data, train_labels, test_data, test_labels, num_epochs=100, batch_size=128):
    """
    Multi-GPU training using pmap. Caps mini-batches at 100, prints loss/acc/time each epoch.
    """
    print("\n[DEBUG] Entering pmap-based training...")
    start_time = time.time()

    num_devices = jax.local_device_count()  # e.g., 2
    print(f"[DEBUG] Number of local devices: {num_devices}")

    num_train = train_data.shape[0]
    # Cap number of mini-batches per epoch at 100
    num_batches = min(num_train // batch_size, 100)

    key = random.PRNGKey(42)
    layer_sizes = [3072, 5000, 4000, 3000, 2000, 1000, 10]
    params = initialize_params(key, layer_sizes)
    opt_state = optimizer.init(params)

    # Replicate across devices
    params = jax.device_put_replicated(params, jax.devices())
    opt_state = jax.device_put_replicated(opt_state, jax.devices())

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Shuffle
        perm = np.random.permutation(num_train)
        train_data = train_data[perm]
        train_labels = train_labels[perm]

        for i in range(num_batches):
            b_start = i * batch_size
            b_end = (i + 1) * batch_size
            x_batch = train_data[b_start:b_end].reshape(batch_size, -1)
            y_batch = train_labels[b_start:b_end]

            # Split for each device
            per_device_batch = batch_size // num_devices
            x_shard = x_batch.reshape(num_devices, per_device_batch, -1)
            y_shard = y_batch.reshape(num_devices, per_device_batch)

            # pmap update
            params, opt_state, loss_val = update_fn(params, opt_state, x_shard, y_shard)

        epoch_time = time.time() - epoch_start
        # Evaluate test metrics using the first device's params
        # We must "extract" param data from device 0 since params is replicated
        # Use jax.tree_map(lambda arr: arr[0], params) to get the actual arrays
        params_host = jax.tree_map(lambda arr: arr[0], params)
        test_loss = cross_entropy_loss(params_host, test_data.reshape(test_data.shape[0], -1), test_labels)
        test_acc = compute_accuracy(params_host, test_data.reshape(test_data.shape[0], -1), test_labels)

        print(f"Epoch {epoch+1:03d} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}% | Epoch Time: {epoch_time:.2f} s")

    total_time = time.time() - start_time
    print(f"[DEBUG] Total training time (pmap): {total_time:.2f} s")

    # Final evaluation
    params_host = jax.tree_map(lambda arr: arr[0], params)
    final_test_loss = cross_entropy_loss(params_host, test_data.reshape(test_data.shape[0], -1), test_labels)
    final_test_acc = compute_accuracy(params_host, test_data.reshape(test_data.shape[0], -1), test_labels)
    print(f"\nFinal Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")

def main():
    script_start = time.time()
    print("\n[DEBUG] JAX devices at script start:", jax.devices())

    # Load CIFAR-10 dataset
    train_data, train_labels, test_data, test_labels = load_cifar10()

    # Train for 100 epochs, capping mini-batches at 100 each epoch, batch_size = 128
    train_mlp_pmap(train_data, train_labels, test_data, test_labels, num_epochs=100, batch_size=128)

    script_end = time.time()
    print(f"\n[DEBUG] Total script execution time: {script_end - script_start:.2f} s")
    print("[DEBUG] Script execution complete!")

if __name__ == "__main__":
    main()
