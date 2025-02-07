import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax

# ------------------------- Data Loading ------------------------- #

def unpickle(file):
    """Unpickle the given file."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10(data_dir="cifar-10-batches-py"):
    """Load CIFAR-10 dataset from binary batches."""
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
    """Initialize weights and biases for a deep feedforward neural network."""
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    
    for i in range(len(layer_sizes) - 1):
        W = random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        params.append((W, b))
    
    return params

# ------------------------- Model Forward Pass ------------------------- #

def batch_norm(x):
    """Apply batch normalization."""
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)  # Normalize inputs

def forward(params, x):
    """Forward pass with batch normalization."""
    for i, (W, b) in enumerate(params[:-1]):
        x = jnp.dot(x, W) + b
        x = batch_norm(x)  # Apply batch normalization
        x = jax.nn.relu(x)  # Activation function

    W_out, b_out = params[-1]
    logits = jnp.dot(x, W_out) + b_out
    return logits

# ------------------------- Loss and Accuracy ------------------------- #

def cross_entropy_loss(params, x, y, l2_lambda=1e-4):
    """Compute cross-entropy loss with L2 regularization."""
    logits = forward(params, x)
    y_one_hot = jax.nn.one_hot(y, num_classes=10)
    ce_loss = -jnp.mean(jnp.sum(y_one_hot * jax.nn.log_softmax(logits), axis=-1))

    # Compute L2 regularization loss
    l2_loss = sum(jnp.sum(W**2) for W, _ in params) * l2_lambda
    return ce_loss + l2_loss  # Combine CE loss and L2 loss

def compute_accuracy(params, x, y):
    """Compute classification accuracy."""
    logits = forward(params, x)
    predictions = jnp.argmax(jax.nn.softmax(logits), axis=-1)  # Apply softmax
    return jnp.mean(predictions == y)

# ------------------------- Optimizer Setup ------------------------- #

def create_optimizer(base_lr=0.01, decay_rate=0.95, decay_steps=10):
    """Create an optimizer with learning rate decay."""
    schedule = optax.exponential_decay(init_value=base_lr, transition_steps=decay_steps, decay_rate=decay_rate)
    return optax.adam(schedule)

optimizer = create_optimizer()  # Global optimizer instance

# ------------------------- Training Step ------------------------- #

@jit
def update(params, opt_state, x, y):
    """Compute gradients and update parameters."""
    loss_value, grads = value_and_grad(cross_entropy_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)  # Use the global optimizer
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

# ------------------------- Training Loop ------------------------- #

def train_mlp(train_data, train_labels, test_data, test_labels, num_epochs=100, batch_size=128):
    """Train the deep feedforward neural network."""
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size
    key = random.PRNGKey(42)

    # Define network architecture
    layer_sizes = [3072, 4000, 3500, 3000, 2500, 1000, 10]  # Increased hidden layer sizes
    params = initialize_params(key, layer_sizes)

    opt_state = optimizer.init(params)  # Correctly initialize optimizer state

    print("\nStarting training...")
    for epoch in range(num_epochs):
        perm = np.random.permutation(num_train)  # Shuffle dataset
        train_data, train_labels = train_data[perm], train_labels[perm]

        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            x_batch, y_batch = train_data[start:end].reshape(batch_size, -1), train_labels[start:end]
            params, opt_state, _ = update(params, opt_state, x_batch, y_batch)

        test_loss = cross_entropy_loss(params, test_data.reshape(test_data.shape[0], -1), test_labels)
        test_acc = compute_accuracy(params, test_data.reshape(test_data.shape[0], -1), test_labels)

        print(f"Epoch {epoch+1:02d} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")

# ------------------------- Main Execution ------------------------- #

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_cifar10()
    print("\nTraining on a single GPU...")
    train_mlp(train_data, train_labels, test_data, test_labels)
