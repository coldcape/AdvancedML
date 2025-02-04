import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp


def check_gpu_availability():
    """Check if JAX detects GPU devices."""
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if gpu_devices:
        print("GPU device(s) available:", gpu_devices)
    else:
        print("No GPU device available; using CPU:", jax.devices())


def unpickle(file):
    """Unpickle the given file."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_cifar100(data_dir="cifar-100-batches-py"):
    """Load CIFAR-100 dataset from binary files."""
    train_batch = unpickle(os.path.join(data_dir, "train"))
    test_batch = unpickle(os.path.join(data_dir, "test"))
    meta = unpickle(os.path.join(data_dir, "meta"))

    train_data = train_batch[b"data"]
    train_labels = np.array(train_batch[b"fine_labels"])  # Fine-grained labels (100 classes)

    test_data = test_batch[b"data"]
    test_labels = np.array(test_batch[b"fine_labels"])

    label_names = [name.decode('utf-8') for name in meta[b"fine_label_names"]]

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    print("CIFAR-100 dataset loaded:")
    print(" Training data shape:", train_data.shape)
    print(" Test data shape:", test_data.shape)
    print(" Number of classes:", len(label_names))

    return train_data, train_labels, test_data, test_labels, label_names


def init_conv_params(key, filter_shape):
    """Initialize parameters for a convolutional layer."""
    k1, _ = jax.random.split(key)
    fan_in = np.prod(filter_shape[:3])
    W = jax.random.normal(k1, filter_shape) * jnp.sqrt(2.0 / fan_in)
    b = jnp.zeros((filter_shape[-1],))
    return W, b


def init_dense_params(key, in_dim, out_dim):
    """Initialize parameters for a dense (fully-connected) layer."""
    k1, _ = jax.random.split(key)
    W = jax.random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
    b = jnp.zeros((out_dim,))
    return W, b


def init_cnn_params(key):
    """Initialize CNN parameters."""
    keys = jax.random.split(key, 4)
    conv1_W, conv1_b = init_conv_params(keys[0], (3, 3, 3, 32))
    conv2_W, conv2_b = init_conv_params(keys[1], (3, 3, 32, 64))
    dense_input_dim = 8 * 8 * 64
    dense1_W, dense1_b = init_dense_params(keys[2], dense_input_dim, 256)
    dense2_W, dense2_b = init_dense_params(keys[3], 256, 100)  # 100 output classes

    return {
        'conv1': (conv1_W, conv1_b),
        'conv2': (conv2_W, conv2_b),
        'dense1': (dense1_W, dense1_b),
        'dense2': (dense2_W, dense2_b)
    }


def cnn_forward(params, x):
    """Forward pass for CNN."""
    conv1_W, conv1_b = params['conv1']
    conv2_W, conv2_b = params['conv2']
    dense1_W, dense1_b = params['dense1']
    dense2_W, dense2_b = params['dense2']

    x = jax.lax.conv_general_dilated(
        x, conv1_W, window_strides=(1, 1),
        padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) + conv1_b
    x = jax.nn.relu(x)

    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding='SAME'
    )

    x = jax.lax.conv_general_dilated(
        x, conv2_W, window_strides=(1, 1),
        padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) + conv2_b
    x = jax.nn.relu(x)

    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding='SAME'
    )

    x = x.reshape((x.shape[0], -1))
    x = jnp.dot(x, dense1_W) + dense1_b
    x = jax.nn.relu(x)

    return jnp.dot(x, dense2_W) + dense2_b


def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss."""
    one_hot = jax.nn.one_hot(labels, num_classes=100)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(one_hot * log_probs, axis=-1).mean()


def compute_loss(params, x, y):
    """Compute loss for a batch."""
    logits = cnn_forward(params, x)
    return cross_entropy_loss(logits, y)


def compute_accuracy(params, x, y):
    """Compute accuracy on a batch."""
    logits = cnn_forward(params, x)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)


@jax.jit
def update(params, x, y, learning_rate):
    """SGD update step."""
    grads = jax.grad(compute_loss)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)


def train_cnn(train_data, train_labels, test_data, test_labels):
    """Train the CNN."""
    num_epochs = 100
    batch_size = 1024
    learning_rate = 0.001
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size

    key = jax.random.PRNGKey(42)
    params = init_cnn_params(key)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        permutation = np.random.permutation(num_train)
        train_data, train_labels = train_data[permutation], train_labels[permutation]

        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            x_batch, y_batch = train_data[start:end], train_labels[start:end]
            params = update(params, x_batch, y_batch, learning_rate)

        train_acc = compute_accuracy(params, train_data[:1000], train_labels[:1000])
        test_acc = compute_accuracy(params, test_data, test_labels)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Acc: {train_acc * 100:.2f}% | "
              f"Test Acc: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    check_gpu_availability()
    train_data, train_labels, test_data, test_labels, label_names = load_cifar100()
    train_cnn(train_data, train_labels, test_data, test_labels)
