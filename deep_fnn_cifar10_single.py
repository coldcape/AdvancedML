"""
deep_FNN_cifar10_singleGPU_best_practices.py

Single-GPU training of a deep feedforward (MLP) network on CIFAR-10 using JAX.
Logs training metrics (loss, accuracy, time) to a CSV.
"""

import os
import csv
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import optax
import argparse

# -------------------------------------------------------------------- #
#                         1) DATA LOADING                              #
# -------------------------------------------------------------------- #
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file_obj:
        data_dict = pickle.load(file_obj, encoding='bytes')
    return data_dict

def load_cifar10(data_dir="cifar-10-batches-py"):
    all_train_images = []
    all_train_labels = []
    for batch_index in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{batch_index}")
        batch_data = load_pickle_file(batch_path)
        all_train_images.append(batch_data[b"data"])
        all_train_labels.extend(batch_data[b"labels"])

    train_data = np.concatenate(all_train_images, axis=0)
    train_labels = np.array(all_train_labels)

    # Reshape
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)

    # Load test
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
    channel_means = train_data.mean(axis=(0, 1, 2))
    channel_stds = train_data.std(axis=(0, 1, 2)) + 1e-7
    train_data_norm = (train_data - channel_means) / channel_stds
    test_data_norm = (test_data - channel_means) / channel_stds
    return train_data_norm, test_data_norm

# -------------------------------------------------------------------- #
#                         2) MODEL & INIT                              #
# -------------------------------------------------------------------- #
def he_init(rng_key, weight_shape):
    num_inputs = weight_shape[0]
    stddev = jnp.sqrt(2.0 / num_inputs)
    return stddev * random.normal(rng_key, weight_shape)

def initialize_parameters(rng_key, network_layer_sizes):
    layer_rng_keys = random.split(rng_key, len(network_layer_sizes) - 1)
    parameters = []
    for i in range(len(network_layer_sizes) - 1):
        in_size = network_layer_sizes[i]
        out_size = network_layer_sizes[i + 1]
        weight_shape = (in_size, out_size)

        weight_matrix = he_init(layer_rng_keys[i], weight_shape)
        bias_vector = jnp.zeros((out_size,), dtype=jnp.float32)
        parameters.append((weight_matrix, bias_vector))
    return parameters

# -------------------------------------------------------------------- #
#                      3) FORWARD / LOSS / METRICS                     #
# -------------------------------------------------------------------- #
def forward_pass(model_params, inputs):
    for (weights, biases) in model_params[:-1]:
        inputs = inputs @ weights + biases
        inputs = jax.nn.relu(inputs)
    final_weights, final_biases = model_params[-1]
    logits = inputs @ final_weights + final_biases
    return logits

def cross_entropy_loss(model_params, inputs, labels):
    logits = forward_pass(model_params, inputs)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))

def compute_accuracy(model_params, inputs, labels):
    logits = forward_pass(model_params, inputs)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# -------------------------------------------------------------------- #
#                   4) OPTIMIZER & TRAIN STEP                          #
# -------------------------------------------------------------------- #
def create_optimizer(base_lr=1e-3, decay_rate=0.98, decay_steps=100):
    lr_schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    return optax.adam(lr_schedule)

optimizer = create_optimizer()

def single_train_step(model_params, optimizer_state, batch_inputs, batch_labels):
    loss_value, grads = value_and_grad(cross_entropy_loss)(model_params, batch_inputs, batch_labels)
    updates, updated_optimizer_state = optimizer.update(grads, optimizer_state, model_params)
    updated_params = optax.apply_updates(model_params, updates)
    return updated_params, updated_optimizer_state, loss_value

jit_train_step = jit(single_train_step)

# -------------------------------------------------------------------- #
#                            5) TRAIN LOOP                             #
# -------------------------------------------------------------------- #
def train_mlp(train_data, train_labels, test_data, test_labels,
              num_epochs=100, batch_size=128, log_csv=None):
    """
    Train the MLP on a single GPU and log epoch-based metrics (loss, accuracy, time) to CSV.
    """
    training_start_time = time.time()
    num_train_samples = train_data.shape[0]
    num_batches = num_train_samples // batch_size

    rng_key = random.PRNGKey(42)
    network_layer_sizes = [3072, 2500, 2000, 1500, 1000, 500, 10]
    model_params = initialize_parameters(rng_key, network_layer_sizes)
    optimizer_state = optimizer.init(model_params)

    setup_time = time.time() - training_start_time
    print(f"Setup time before training: {setup_time:.2f} seconds")

    # Prepare CSV logging if log_csv is provided
    if log_csv:
        with open(log_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "epoch_time_s", "train_loss (last batch)", "test_loss", "test_accuracy"])

    print("\nStarting training on single GPU...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Shuffle each epoch
        shuffle_indices = np.random.permutation(num_train_samples)
        train_data = train_data[shuffle_indices]
        train_labels = train_labels[shuffle_indices]

        # Train loop
        last_batch_loss = 0.0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_inputs = train_data[start_idx:end_idx].reshape(batch_size, -1)
            batch_labels_slice = train_labels[start_idx:end_idx]

            model_params, optimizer_state, train_loss = jit_train_step(
                model_params,
                optimizer_state,
                batch_inputs,
                batch_labels_slice
            )
            last_batch_loss = train_loss  # keep track of final batch's loss in the epoch

        # Evaluate
        epoch_time = time.time() - epoch_start_time
        test_inputs = test_data.reshape(test_data.shape[0], -1)
        test_loss = cross_entropy_loss(model_params, test_inputs, test_labels)
        test_accuracy = compute_accuracy(model_params, test_inputs, test_labels)

        print(f"Epoch {epoch+1:02d}/{num_epochs} "
              f"| Test Loss: {test_loss:.4f} "
              f"| Test Acc: {test_accuracy*100:.2f}% "
              f"(Epoch time: {epoch_time:.2f}s)")

        # Write epoch metrics to CSV
        if log_csv:
            with open(log_csv, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    f"{epoch_time:.2f}",
                    f"{float(last_batch_loss):.4f}",
                    f"{float(test_loss):.4f}",
                    f"{float(test_accuracy)*100:.2f}"
                ])

    total_training_time = time.time() - training_start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds")

# -------------------------------------------------------------------- #
#                        6) MAIN EXECUTION                             #
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

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

    # 3) Train
    training_start_time = time.time()
    train_mlp(train_data, train_labels, test_data, test_labels,
              num_epochs=100,
              batch_size=128,
              log_csv=args.log_csv)
    training_end_time = time.time()
    print(f"\nTime spent in train_mlp function: {training_end_time - training_start_time:.2f} seconds")

    script_end_time = time.time()
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")
