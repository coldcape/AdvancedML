import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from datetime import datetime
import csv
import time  # ← add this at the top if not already

# ---- Load CIFAR-10 Python version ----
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

# ---- CNN Model ----
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2))
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# ---- Training and Evaluation ----
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        logits = state.apply_fn(params, batch_x)
        one_hot = jax.nn.one_hot(batch_y, 10)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

@jax.jit
def eval_step(state, batch_x):
    logits = state.apply_fn(state.params, batch_x)
    return jnp.argmax(logits, axis=-1)

def compute_accuracy(state, x_test, y_test, batch_size=100):
    correct = 0
    total = 0
    for i in range(0, len(x_test), batch_size):
        batch_x = jnp.array(x_test[i:i+batch_size])
        batch_y = y_test[i:i+batch_size]
        preds = eval_step(state, batch_x)
        correct += np.sum(np.array(preds) == np.array(batch_y))
        total += len(batch_y)
    return correct / total

# ---- Main ----
import time  # ← add this at the top if not already

def main():
    # Dataset
    (x_train, y_train), (x_test, y_test) = load_cifar10('cifar-10-batches-py')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Output directory and CSV logging
    os.makedirs("experiments", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"experiments/{timestamp}_single_gpu_training_log.csv"

    rng = jax.random.PRNGKey(0)
    model = CNN()
    state = create_train_state(rng, model, learning_rate=0.001)

    total_start_time = time.time()

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'accuracy', 'epoch_time_seconds'])

        batch_size = 128
        for epoch in range(25):
            epoch_start_time = time.time()

            # Shuffle each epoch
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]

            for i in range(0, len(x_train), batch_size):
                batch_x = jnp.array(x_train[i:i+batch_size])
                batch_y = jnp.array(y_train[i:i+batch_size])
                state = train_step(state, batch_x, batch_y)

            acc = compute_accuracy(state, x_test, y_test)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} complete - Accuracy: {acc:.4f} - Time: {epoch_time:.2f}s")
            writer.writerow([epoch + 1, acc, epoch_time])

    total_time = time.time() - total_start_time
    print(f"Training complete - Total Time: {total_time:.2f}s")

    # Optional: log total time to a separate file
    with open(f"experiments/{timestamp}_single_GPU_total_training_time.txt", "w") as tf:
        tf.write(f"Total training time: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
