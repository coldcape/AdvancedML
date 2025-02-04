#!/usr/bin/env python
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# Data Loading and Augmentation for CIFAR-100
# -----------------------------------------------------------------------------
def unpickle(file):
    """Unpickle a file and return its dictionary."""
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load_cifar100(data_dir="cifar-100-batches-py"):
    """
    Load the CIFAR-100 dataset.
    
    Expects three files in the folder:
      - train: training set
      - test: testing set
      - meta: metadata (with fine label names)
    
    Returns:
      train_data, train_labels, test_data, test_labels, label_names
    """
    train_batch = unpickle(os.path.join(data_dir, "train"))
    test_batch = unpickle(os.path.join(data_dir, "test"))
    meta = unpickle(os.path.join(data_dir, "meta"))
    
    train_data = train_batch[b"data"]
    train_labels = np.array(train_batch[b"fine_labels"])
    test_data = test_batch[b"data"]
    test_labels = np.array(test_batch[b"fine_labels"])
    
    label_names = [name.decode('utf-8') for name in meta[b"fine_label_names"]]
    
    # Reshape flat images into (N, 32, 32, 3) in NHWC format
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Normalize pixel values to [0, 1]
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0
    
    print("CIFAR-100 dataset loaded:")
    print(" Training data shape:", train_data.shape)
    print(" Test data shape:", test_data.shape)
    print(" Number of classes:", len(label_names))
    
    return train_data, train_labels, test_data, test_labels, label_names


def augment_batch(images, key):
    """
    Apply data augmentation to a batch of images.
    
    Currently applies random horizontal flip.
    
    Args:
      images: input batch of shape (batch, H, W, C)
      key: JAX PRNG key.
    
    Returns:
      Augmented images.
    """
    # Generate random booleans for each image (flip with probability 0.5)
    flip_key, _ = jax.random.split(key)
    flip = jax.random.bernoulli(flip_key, p=0.5, shape=(images.shape[0],))
    
    def flip_image(img, do_flip):
        # If do_flip is True, reverse the image horizontally; otherwise, keep unchanged.
        return jnp.where(do_flip, img[:, ::-1, :], img)
    
    # Vectorize the flipping function over the batch.
    return jax.vmap(flip_image)(images, flip)


# -----------------------------------------------------------------------------
# Convolution, Dense Layers, and Parameter Initialization
# -----------------------------------------------------------------------------
def conv2d(x, W, b, stride=1, padding="SAME"):
    """
    Perform a 2D convolution with bias addition.
    
    Args:
      x: input tensor with shape (batch, H, W, C_in)
      W: convolution filter of shape (filter_height, filter_width, C_in, C_out)
      b: bias vector (C_out,)
      stride: convolution stride (assumed same for height and width)
      padding: "SAME" or "VALID"
    
    Returns:
      The convolved output plus bias.
    """
    conv = jax.lax.conv_general_dilated(
        x, W,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return conv + b


def init_conv_params(key, filter_shape):
    """
    Initialize convolutional layer parameters using He initialization.
    
    Args:
      key: PRNG key.
      filter_shape: (filter_height, filter_width, C_in, C_out)
    
    Returns:
      Tuple (W, b)
    """
    k1, _ = jax.random.split(key)
    fan_in = np.prod(filter_shape[:3])
    W = jax.random.normal(k1, filter_shape) * jnp.sqrt(2.0 / fan_in)
    b = jnp.zeros((filter_shape[-1],))
    return W, b


def init_dense_params(key, in_dim, out_dim):
    """
    Initialize parameters for a dense (fully-connected) layer.
    
    Args:
      key: PRNG key.
      in_dim: input dimension.
      out_dim: output dimension.
    
    Returns:
      Tuple (W, b)
    """
    k1, _ = jax.random.split(key)
    W = jax.random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
    b = jnp.zeros((out_dim,))
    return W, b


# -----------------------------------------------------------------------------
# Residual Block Functions
# -----------------------------------------------------------------------------
def init_residual_block_params(key, in_channels, out_channels, use_projection):
    """
    Initialize parameters for one residual block.
    
    Two 3x3 convolutional layers are used. If use_projection is True,
    a 1x1 convolution is used in the shortcut to match dimensions.
    
    Args:
      key: PRNG key.
      in_channels: number of input channels.
      out_channels: number of output channels.
      use_projection: bool; if True, a projection layer is added.
    
    Returns:
      Dictionary of parameters.
    """
    keys = jax.random.split(key, 3 if use_projection else 2)
    params = {}
    params["conv1"] = init_conv_params(keys[0], (3, 3, in_channels, out_channels))
    params["conv2"] = init_conv_params(keys[1], (3, 3, out_channels, out_channels))
    if use_projection:
        params["proj"] = init_conv_params(keys[2], (1, 1, in_channels, out_channels))
    return params


def residual_block(params, x, stride):
    """
    Forward pass for a residual block.
    
    Args:
      params: dictionary containing "conv1", "conv2" (and optionally "proj")
      x: input tensor.
      stride: stride for the first convolution.
    
    Returns:
      Output tensor after applying the block.
    """
    out = conv2d(x, *params["conv1"], stride=stride, padding="SAME")
    out = jax.nn.relu(out)
    out = conv2d(out, *params["conv2"], stride=1, padding="SAME")
    
    # Shortcut connection: apply projection if necessary.
    if "proj" in params:
        shortcut = conv2d(x, *params["proj"], stride=stride, padding="SAME")
    else:
        shortcut = x
    
    return jax.nn.relu(out + shortcut)


# -----------------------------------------------------------------------------
# ResNet-18 Model Initialization and Forward Pass with Dropout
# -----------------------------------------------------------------------------
def init_resnet18_params(key, num_classes=100):
    """
    Initialize parameters for a ResNet-18–like model.
    
    Architecture for CIFAR-100:
      - Initial 3x3 conv (64 filters)
      - Four groups of residual blocks:
          Group 1: 2 blocks, 64 filters, no downsampling.
          Group 2: 2 blocks, 128 filters, first block downsamples.
          Group 3: 2 blocks, 256 filters, first block downsamples.
          Group 4: 2 blocks, 512 filters, first block downsamples.
      - Global average pooling and a final FC layer.
    
    Returns:
      Dictionary of parameters.
    """
    params = {}
    keys = jax.random.split(key, 20)
    key_idx = 0

    # Initial convolution layer.
    params["conv_init"] = init_conv_params(keys[key_idx], (3, 3, 3, 64))
    key_idx += 1

    # Residual layers.
    params["layer1"] = []
    for i in range(2):
        block_params = init_residual_block_params(keys[key_idx], 64, 64, use_projection=False)
        key_idx += 1
        params["layer1"].append(block_params)

    params["layer2"] = []
    for i in range(2):
        if i == 0:
            block_params = init_residual_block_params(keys[key_idx], 64, 128, use_projection=True)
        else:
            block_params = init_residual_block_params(keys[key_idx], 128, 128, use_projection=False)
        key_idx += 1
        params["layer2"].append(block_params)

    params["layer3"] = []
    for i in range(2):
        if i == 0:
            block_params = init_residual_block_params(keys[key_idx], 128, 256, use_projection=True)
        else:
            block_params = init_residual_block_params(keys[key_idx], 256, 256, use_projection=False)
        key_idx += 1
        params["layer3"].append(block_params)

    params["layer4"] = []
    for i in range(2):
        if i == 0:
            block_params = init_residual_block_params(keys[key_idx], 256, 512, use_projection=True)
        else:
            block_params = init_residual_block_params(keys[key_idx], 512, 512, use_projection=False)
        key_idx += 1
        params["layer4"].append(block_params)

    # Fully connected layer parameters (after global average pooling).
    fc_input_dim = 512  # Global average pooling produces (batch, 512)
    params["fc"] = init_dense_params(keys[key_idx], fc_input_dim, num_classes)
    key_idx += 1

    return params


def resnet18_forward(params, x, dropout_rate=0.5, key=None, is_training=True):
    """
    Forward pass for the ResNet-18–like network.
    
    Args:
      params: dictionary of model parameters.
      x: input tensor (batch, H, W, C)
      dropout_rate: probability of dropping neurons (applied after global average pooling).
      key: PRNG key for dropout.
      is_training: if True, dropout is applied.
    
    Returns:
      Logits (batch, num_classes)
    """
    # Initial convolution and ReLU.
    x = conv2d(x, *params["conv_init"], stride=1, padding="SAME")
    x = jax.nn.relu(x)

    # Residual layers.
    for block_params in params["layer1"]:
        x = residual_block(block_params, x, stride=1)
    for i, block_params in enumerate(params["layer2"]):
        stride = 2 if i == 0 else 1
        x = residual_block(block_params, x, stride=stride)
    for i, block_params in enumerate(params["layer3"]):
        stride = 2 if i == 0 else 1
        x = residual_block(block_params, x, stride=stride)
    for i, block_params in enumerate(params["layer4"]):
        stride = 2 if i == 0 else 1
        x = residual_block(block_params, x, stride=stride)

    # Global average pooling (average over H and W dimensions).
    x = x.mean(axis=(1, 2))

    # Apply dropout if in training mode.
    if is_training and key is not None:
        dropout_key, _ = jax.random.split(key)
        keep_prob = 1.0 - dropout_rate
        mask = jax.random.bernoulli(dropout_key, p=keep_prob, shape=x.shape)
        x = x * mask / keep_prob

    # Final fully connected layer.
    logits = jnp.dot(x, params["fc"][0]) + params["fc"][1]
    return logits


# -----------------------------------------------------------------------------
# Loss, Accuracy, and Training Step Functions with Weight Decay
# -----------------------------------------------------------------------------
def cross_entropy_loss(logits, labels):
    """
    Compute cross-entropy loss.
    
    Args:
      logits: unnormalized log probabilities (batch, num_classes)
      labels: integer class labels (batch,)
    
    Returns:
      Scalar loss value.
    """
    one_hot = jax.nn.one_hot(labels, num_classes=100)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


def compute_accuracy(params, x, y, dropout_rate=0.5, key=None):
    """
    Compute accuracy on a batch.
    
    Args:
      params: model parameters.
      x: input images.
      y: true labels.
      dropout_rate: dropout rate (set is_training=False for evaluation).
      key: PRNG key (not used during evaluation).
    
    Returns:
      Accuracy (scalar).
    """
    logits = resnet18_forward(params, x, dropout_rate, key, is_training=False)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)


@jax.jit
def update(params, x, y, learning_rate, weight_decay, dropout_key):
    """
    Perform one SGD update step with weight decay (L2 regularization).
    
    Args:
      params: current model parameters.
      x: input batch.
      y: labels.
      learning_rate: learning rate for SGD.
      weight_decay: L2 penalty coefficient.
      dropout_key: PRNG key for dropout.
    
    Returns:
      Updated parameters.
    """
    # Compute gradients of the loss (including dropout randomness).
    loss_fn = lambda p: cross_entropy_loss(
        resnet18_forward(p, x, dropout_rate=0.5, key=dropout_key, is_training=True), y)
    grads = jax.grad(loss_fn)(params)
    
    # Update parameters with weight decay.
    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * (g + weight_decay * p), params, grads)
    return new_params


def schedule_lr(initial_lr, epoch, decay_rate=0.5, decay_epochs=20):
    """
    Simple exponential decay scheduler for the learning rate.
    
    Args:
      initial_lr: starting learning rate.
      epoch: current epoch number.
      decay_rate: factor by which to decay.
      decay_epochs: decay every this many epochs.
    
    Returns:
      Adjusted learning rate.
    """
    return initial_lr * (decay_rate ** (epoch // decay_epochs))


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_resnet(train_data, train_labels, test_data, test_labels, num_epochs=64):
    """
    Train the ResNet-18–like model on CIFAR-100.
    
    Args:
      train_data: training images.
      train_labels: training labels.
      test_data: testing images.
      test_labels: testing labels.
      num_epochs: number of training epochs.
    """
    batch_size = 128
    initial_lr = 0.001
    weight_decay = 1e-4
    num_train = train_data.shape[0]
    num_batches = num_train // batch_size

    # Initialize model parameters.
    key = jax.random.PRNGKey(42)
    params = init_resnet18_params(key, num_classes=100)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Adjust learning rate using scheduler.
        lr = schedule_lr(initial_lr, epoch)
        
        # Shuffle training data.
        permutation = np.random.permutation(num_train)
        train_data_epoch = train_data[permutation]
        train_labels_epoch = train_labels[permutation]
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = train_data_epoch[start:end]
            y_batch = train_labels_epoch[start:end]
            
            # Data augmentation: apply random horizontal flip.
            aug_key = jax.random.fold_in(key, epoch * num_batches + i)
            x_batch = augment_batch(x_batch, aug_key)
            
            # Generate a dropout key for this batch.
            dropout_key = jax.random.fold_in(key, epoch * num_batches + i + 1000)
            params = update(params, x_batch, y_batch, lr, weight_decay, dropout_key)
        
        # Evaluate on a subset of training data and the full test set.
        eval_key = jax.random.PRNGKey(epoch)
        train_acc = compute_accuracy(params, train_data[:1000], train_labels[:1000],
                                     dropout_rate=0.5, key=eval_key)
        test_acc = compute_accuracy(params, test_data, test_labels, dropout_rate=0.5, key=eval_key)
        print(f"Epoch {epoch+1:02d} | LR: {lr:.6f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")


# -----------------------------------------------------------------------------
# GPU Check and Main Execution Block
# -----------------------------------------------------------------------------
def check_gpu_availability():
    """Print available GPU devices according to JAX."""
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if gpu_devices:
        print("GPU device(s) available:", gpu_devices)
    else:
        print("No GPU device available; using CPU:", jax.devices())


if __name__ == "__main__":
    check_gpu_availability()
    
    # Load CIFAR-100 data.
    train_data, train_labels, test_data, test_labels, label_names = load_cifar100()
    
    # Train the network.
    train_resnet(train_data, train_labels, test_data, test_labels)
