"""
deep_fnn_cifar10_pmap.py

Multi-GPU (2 GPUs) training of a deep feedforward (MLP) network on CIFAR-10 using JAX pmap,
logging epoch-based metrics to CSV.
"""

import os
import csv
import time
import pickle  # <-- IMPORTANT: needed for load_pickle_file
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax
from functools import partial
import argparse

# -------------------------------------------------------------------- #
#                         1) DATA LOADING                              #
# -------------------------------------------------------------------- #
def load_pickle_file(file_path):
    """Load a pickled file (binary) from CIFAR-10 dataset."""
    with open(file_path, 'rb') as file_obj:
        data_dict = pickle.load(file_obj, encoding='bytes')
    return data_dict

def load_cifar10(data_dir="cifar-10-batches-py"):
    """
    Load the CIFAR-10 dataset from disk and return training/testing data and labels.
    """
    all_train_images = []
    all_train_labels = []
    for batch_index in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{batch_index}")
        batch_data = load_pickle_file(batch_path)
        all_train_images.append(batch_data[b"data"])
        all_train_labels.extend(batch_data[b"labels"])

    train_data = np.concatenate(all_train_images, axis=0)
    train_labels = np.array(all_train_labels)

    # Reshape to (N, 32, 32, 3)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)

    # Load test set
    test_batch_path = os.path.join(data_dir, "test_batch")
    test_batch = load_pickle_file(test_batch_path)
    test_data = test_batch[b"data"]
    test_labels = np.array(test_batch[b"labels"])
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)

    print("CIFAR-10 dataset loaded.")
    print(" Training data shape:", train_data.shape)
    print(" Test data shape:", test_data.shape)
    return train_data, train_labels, test_data, test_labels

def normalize_cifar10(train_data, test_data):
    """Normalize CIFAR-10 by subtracting mean and dividing by std (computed from training)."""
    channel_means = train_data.mean(axis=(0, 1, 2))
    channel_stds = train_data.std(axis=(0, 1, 2)) + 1e-7
    train_data_norm = (train_data - channel_means) / channel_stds
    test_data_norm = (test_data - channel_means) / channel_stds
    return train_data_norm, test_data_norm

# -------------------------------------------------------------------- #
#                         2) MODEL & INIT                              #
# -------------------------------------------------------------------- #
def he_init(rng_key, weight_shape):
    """He (Kaiming) initialization for ReLU layers."""
    num_inputs = weight_shape[0]
    stddev = jnp.sqrt(2.0 / num_inputs)
    return stddev * random.normal(rng_key, weight_shape)

def initialize_parameters(rng_key, network_layer_sizes):
    """Initialize weights and biases for each layer."""
    layer_rng_keys = random.split(rng_key, len(network_layer_sizes) - 1)
    parameters = []
    for i in range(len(network_layer_sizes) - 1):
        in_size = network_layer_sizes[i]
        out_size = network_layer_sizes[i + 1]
        weight_shape = (in_size, out_size)

        W = he_init(layer_rng_keys[i], weight_shape)
        b = jnp.zeros((out_size,), dtype=jnp.float32)
        parameters.append((W, b))
    return parameters

# -------------------------------------------------------------------- #
#                      3) FORWARD / LOSS / METRICS                     #
# -------------------------------------------------------------------- #
def forward_pass(model_params, inputs):
    """Forward pass of the MLP."""
    for (weights, biases) in model_params[:-1]:
        inputs = inputs @ weights + biases
        inputs = jax.nn.relu(inputs)
    final_weights, final_biases = model_params[-1]
    logits = inputs @ final_weights + final_biases
    return logits

def cross_entropy_loss(model_params, inputs, labels):
    """Compute cross-entropy loss."""
    logits = forward_pass(model_params, inputs)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))

def compute_accuracy(model_params, inputs, labels):
    """Compute classification accuracy."""
    logits = forward_pass(model_params, inputs)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)

# -------------------------------------------------------------------- #
#                   4) OPTIMIZER & TRAIN STEP                          #
# -------------------------------------------------------------------- #
def create_optimizer(base_lr=1e-3, decay_rate=0.98, decay_steps=100):
    """Adam optimizer with exponential decay learning rate schedule."""
    lr_schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    return optax.adam(lr_schedule)

optimizer = create_optimizer()

# Parallelized train step with pmap
@partial(jax.pmap, axis_name='num_devices')
def parallel_train_step(model_params, optimizer_state, batch_inputs, batch_labels):
    def loss_fn(params, x, y):
        return cross_entropy_loss(params, x, y)

    loss_value, grads = value_and_grad(loss_fn)(model_params, batch_inputs, batch_labels)
    # Average gradients across devices
    grads = jax.tree_map(lambda g: jax.lax.pmean(g, axis_name='num_devices'), grads)

    updates, new_optimizer_state = optimizer.update(grads, optimizer_state, model_params)
    new_model_params = optax.apply_updates(model_params, updates)

    # Average loss across devices
    avg_loss = jax.lax.pmean(loss_value, axis_name='num_devices')
    return new_model_params, new_optimizer_state, avg_loss

# Parallelized eval step with pmap
@partial(jax.pmap, axis_name='num_devices')
def parallel_eval_step(model_params, batch_inputs, batch_labels):
    """Compute mean loss & accuracy across devices."""
    logits = forward_pass(model_params, batch_inputs)
    one_hot_labels = jax.nn.one_hot(batch_labels, num_classes=10)
    log_probs = jax.nn.log_softmax(logits)

    loss_value = -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == batch_labels)

    avg_loss = jax.lax.pmean(loss_value, axis_name='num_devices')
    avg_acc = jax.lax.pmean(accuracy, axis_name='num_devices')
    return avg_loss, avg_acc

# -------------------------------------------------------------------- #
#                            5) DATASET HELPERS                        #
# -------------------------------------------------------------------- #
def create_sharded_dataset(data, labels, batch_size, num_devices, shuffle=True):
    """
    Generates sharded mini-batches for pmap across multiple GPUs.
    Each mini-batch is shaped: (num_devices, local_batch_size, ...) 
    """
    num_samples = data.shape[0]
    local_batch_size = batch_size // num_devices
    num_batches = num_samples // batch_size

    if shuffle:
        perm = np.random.permutation(num_samples)
        data = data[perm]
        labels = labels[perm]

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch_inputs = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        # Flatten: (batch_size, 32, 32, 3) -> (batch_size, 3072)
        batch_inputs = batch_inputs.reshape(batch_size, -1)

        # Shard: (num_devices, local_batch_size, 3072)
        inputs_shard = batch_inputs.reshape(num_devices, local_batch_size, -1)
        labels_shard = batch_labels.reshape(num_devices, local_batch_size)
        yield inputs_shard, labels_shard

def create_sharded_eval_dataset(data, labels, batch_size, num_devices):
    """Non-shuffled version for evaluation."""
    return create_sharded_dataset(data, labels, batch_size, num_devices, shuffle=False)

# -------------------------------------------------------------------- #
#                            6) TRAIN LOOP                             #
# -------------------------------------------------------------------- #
def train_mlp(train_data, train_labels, test_data, test_labels,
              num_epochs=100, batch_size=128, log_csv=None):
    """
    Train the MLP using pmap across 2+ GPUs and log epoch-based metrics to CSV if log_csv is provided.
    """
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices < 2:
        raise ValueError("This script requires at least 2 GPUs.")

    training_start_time = time.time()
    rng_key = random.PRNGKey(42)
    network_layer_sizes = [3072, 2500, 2000, 1500, 1000, 500, 10]

    # Initialize params/optimizer (single-device)
    model_params = initialize_parameters(rng_key, network_layer_sizes)
    optimizer_state = optimizer.init(model_params)

    # Replicate them to each device
    model_params = jax.device_put_replicated(model_params, devices)
    optimizer_state = jax.device_put_replicated(optimizer_state, devices)

    setup_time = time.time() - training_start_time
    print(f"Setup time before training: {setup_time:.2f} seconds")

    print(f"\nStarting training on {num_devices} GPUs...")

    # Prepare CSV logging if a path is specified
    if log_csv:
        with open(log_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "epoch_time_s", "train_loss (last shard)", "test_loss", "test_accuracy"])

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Create a sharded dataset loader
        train_data_loader = create_sharded_dataset(train_data, train_labels, batch_size, num_devices, shuffle=True)
        last_shard_loss = 0.0

        # Training loop for all batches in this epoch
        for inputs_shard, labels_shard in train_data_loader:
            model_params, optimizer_state, train_loss = parallel_train_step(
                model_params,
                optimizer_state,
                inputs_shard,
                labels_shard
            )
            # train_loss is shape (num_devices,) with identical values, so use [0]
            last_shard_loss = float(train_loss[0])

        # Parallel Evaluation
        test_data_loader = create_sharded_eval_dataset(test_data, test_labels, batch_size, num_devices)
        total_loss = 0.0
        total_acc = 0.0
        num_eval_batches = 0

        for test_inputs_shard, test_labels_shard in test_data_loader:
            loss_val, acc_val = parallel_eval_step(model_params, test_inputs_shard, test_labels_shard)
            total_loss += float(loss_val[0])
            total_acc += float(acc_val[0])
            num_eval_batches += 1

        # Averages
        avg_test_loss = total_loss / num_eval_batches
        avg_test_acc = total_acc / num_eval_batches
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1:02d}/{num_epochs} "
              f"| Test Loss: {avg_test_loss:.4f} "
              f"| Test Acc: {avg_test_acc*100:.2f}% "
              f"(Epoch time: {epoch_time:.2f}s)")

        # Log epoch metrics to CSV
        if log_csv:
            with open(log_csv, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    f"{epoch_time:.2f}",
                    f"{last_shard_loss:.4f}",
                    f"{avg_test_loss:.4f}",
                    f"{avg_test_acc*100:.2f}"
                ])

    total_training_time = time.time() - training_start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds")

# -------------------------------------------------------------------- #
#                        7) MAIN EXECUTION                             #
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, default=None,
                        help="Path to CSV file for training metrics.")
    args = parser.parse_args()

    script_start_time = time.time()

    # 1) Load CIFAR-10
    train_data, train_labels, test_data, test_labels = load_cifar10()

    # 2) Normalize
    train_data, test_data = normalize_cifar10(train_data, test_data)
    data_loading_end_time = time.time()
    print(f"\nData loading + normalization time: {data_loading_end_time - script_start_time:.2f} seconds")

    # 3) Train using pmap
    training_start_time = time.time()
    train_mlp(train_data, train_labels, test_data, test_labels,
              num_epochs=100,
              batch_size=256,
              log_csv=args.log_csv)
    training_end_time = time.time()
    print(f"\nTime spent in train_mlp function: {training_end_time - training_start_time:.2f} seconds")

    script_end_time = time.time()
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")
