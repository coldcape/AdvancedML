import os
import pickle
import time
import csv
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from datetime import datetime

# ----------------- CIFAR-10 Loading -----------------
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b'labels'])
    return data, labels

def load_cifar10(path):
    xs, ys = [], []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(os.path.join(path, f'data_batch_{i}'))
        xs.append(data)
        ys.append(labels)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    x_test, y_test = load_cifar10_batch(os.path.join(path, 'test_batch'))
    return (x_train, y_train), (x_test, y_test)

# ----------------- CNN Model -----------------
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2))
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# ----------------- Utility Functions -----------------
def create_train_state(rng, model, learning_rate):
    dummy_input = jnp.ones([1, 32, 32, 3])
    params = model.init(rng, dummy_input)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.pmap
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        one_hot = jax.nn.one_hot(y, 10)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

@jax.pmap
def eval_step(state, x):
    logits = state.apply_fn(state.params, x)
    return jnp.argmax(logits, axis=-1)

def shard(data, num_devices):
    return data.reshape((num_devices, -1) + data.shape[1:])

def compute_accuracy(state, x_test, y_test, num_devices, batch_size=128):
    correct = 0
    total = 0
    for i in range(0, len(x_test), batch_size):
        if i + batch_size > len(x_test):
            continue
        batch_x = jnp.array(x_test[i:i+batch_size])
        batch_y = y_test[i:i+batch_size]
        preds = eval_step(state, shard(batch_x, num_devices))
        preds = np.array(preds).reshape(-1)
        correct += np.sum(preds == batch_y)
        total += len(batch_y)
    return correct / total

# ----------------- Main Training -----------------
def main():
    print("JAX devices:", jax.devices())
    num_devices = len(jax.devices())
    assert num_devices == 2, "Expected 2 GPUs for pmap."

    (x_train, y_train), (x_test, y_test) = load_cifar10('cifar-10-batches-py')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    os.makedirs("experiments", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"experiments/{timestamp}_multi_gpu_training_log.csv"

    rng = jax.random.PRNGKey(0)
    model = CNN()
    state = create_train_state(rng, model, learning_rate=0.001)

    # Replicate model state across GPUs
    state = jax.device_put_replicated(state, jax.devices()[:num_devices])
    print("Model state replicated to devices.")

    total_start = time.time()

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "epoch_time_sec"])

        batch_size = 256
        assert batch_size % num_devices == 0

        for epoch in range(1, 26):
            start = time.time()
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]

            for i in range(0, len(x_train), batch_size):
                if i + batch_size > len(x_train):
                    continue  # Drop last incomplete batch

                batch_x = jnp.array(x_train[i:i+batch_size])
                batch_y = jnp.array(y_train[i:i+batch_size])

                sh_x = shard(batch_x, num_devices)
                sh_y = shard(batch_y, num_devices)

                if epoch == 1 and i == 0:  # Print once for verification
                    print("Sharded x shape:", sh_x.shape)
                    print("Sharded y shape:", sh_y.shape)

                state = train_step(state, sh_x, sh_y)

            acc = compute_accuracy(state, x_test, y_test, num_devices)
            epoch_time = time.time() - start
            print(f"Epoch {epoch} complete - Accuracy: {acc:.4f} - Time: {epoch_time:.2f}s")
            writer.writerow([epoch, acc, epoch_time])

    total_time = time.time() - total_start
    with open(f"experiments/{timestamp}_mult_gpu_total_time.txt", 'w') as tf:
        tf.write(f"Total training time: {total_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
