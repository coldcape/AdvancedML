import os
import pickle
import time  # For timing
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax

# ------------------------- Data Loading ------------------------- #

def unpickle(file):
    """
    Unpickle the given file using Python's pickle.
    This loads serialized Python objects from a binary file.

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

    Steps:
    1. Read training batches (data_batch_1 to data_batch_5).
    2. Concatenate them into a single training set (train_data, train_labels).
    3. Read the test batch (test_batch).
    4. Reshape data into (N, 32, 32, 3) format, where N is the number of images.
    5. Normalize image pixel values to the range [0,1] by dividing by 255.0.

    Args:
        data_dir (str): The directory where CIFAR-10 batch files are located.

    Returns:
        train_data (np.ndarray): Training images of shape (50000, 32, 32, 3).
        train_labels (np.ndarray): Training labels of shape (50000,).
        test_data (np.ndarray): Test images of shape (10000, 32, 32, 3).
        test_labels (np.ndarray): Test labels of shape (10000,).
    """
    train_data_list, train_labels_list = [], []
    # CIFAR-10 has 5 training batches, each containing 10,000 images
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        train_data_list.append(batch[b"data"])
        train_labels_list.extend(batch[b"labels"])

    # Concatenate all training batches into one large array
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.array(train_labels_list)

    # Reshape from (N, 3072) to (N, 3, 32, 32) and then transpose to (N, 32, 32, 3)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    # Normalize pixel values from [0,255] to [0,1]
    train_data = train_data.astype(np.float32) / 255.0

    # Load the test set
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

    We create a parameter tuple (W, b, gamma, beta) for each layer, except
    the last layer is also given gamma and beta (though not used in forward
    if we skip batch norm on the last layer). You could adapt this to omit
    them for the last layer if desired.

    Args:
        key (jax.random.PRNGKey): Random number generator key for reproducibility.
        layer_sizes (list): Sizes of each layer (including input and output).
                            Example: [3072, 5000, 4000, 3000, 2000, 1000, 10].

    Returns:
        params (list): A list of tuples [(W, b, gamma, beta), ...] for each layer.
    """
    params = []
    # We split the random key into (number_of_layers - 1) keys, so each layer
    # has a unique key for initialization.
    keys = random.split(key, len(layer_sizes) - 1)
    
    for i in range(len(layer_sizes) - 1):
        # Weight matrix: shape (in_features, out_features).
        # He (Kaiming) initialization: multiply by sqrt(2.0 / fan_in).
        W = random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        
        # Bias vector: shape (out_features,).
        b = jnp.zeros((layer_sizes[i+1],))

        # Batch Norm parameters: gamma (scale), beta (shift). 
        # Initialized so that initially, BN does nothing (gamma=1, beta=0).
        gamma = jnp.ones((layer_sizes[i+1],))
        beta = jnp.zeros((layer_sizes[i+1],))

        params.append((W, b, gamma, beta))
    
    return params


# ------------------------- Model Forward Pass ------------------------- #

def batch_norm(x, gamma, beta):
    """
    Apply batch normalization to the input x, using the learnable parameters gamma and beta.

    Batch Normalization normalizes the input so that it has mean ~ 0 and variance ~ 1,
    computed across the batch dimension. Then it rescales and shifts using gamma and beta.

    Args:
        x (jnp.ndarray): Input data of shape (batch_size, features).
        gamma (jnp.ndarray): Scale parameter of shape (features,).
        beta (jnp.ndarray): Shift parameter of shape (features,).

    Returns:
        (jnp.ndarray): The batch-normalized output of the same shape as x.
    """
    # Compute mean and variance across the batch (axis=0).
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)

    # Normalize x to have zero mean and unit variance.
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)

    # Scale and shift by gamma and beta, which are learnable parameters.
    return gamma * x_norm + beta

def forward(params, x):
    """
    Perform the forward pass through the MLP, using batch normalization and ReLU activations.

    Args:
        params (list): List of parameter tuples (W, b, gamma, beta) for each layer.
        x (jnp.ndarray): Input data of shape (batch_size, input_features).

    Returns:
        logits (jnp.ndarray): Output of the final layer (batch_size, num_classes).
    """
    # Iterate through all but the last layer to apply linear transform, batch norm, and ReLU.
    for (W, b, gamma, beta) in params[:-1]:
        # Linear transformation: xW + b
        x = jnp.dot(x, W) + b

        # Apply batch normalization (helps stabilize and accelerate training)
        x = batch_norm(x, gamma, beta)

        # ReLU activation, a common choice for hidden layers.
        x = jax.nn.relu(x)

    # For the final layer, we only apply the linear transform.
    # Typically, we do not apply BN or ReLU here if it's for classification.
    W_out, b_out, _, _ = params[-1]
    logits = jnp.dot(x, W_out) + b_out

    return logits


# ------------------------- Loss and Accuracy ------------------------- #

def cross_entropy_loss(params, x, y, l2_lambda=5e-5):
    """
    Compute the cross-entropy loss (with softmax) and add L2 regularization on the weights.

    Args:
        params (list): Model parameters [(W, b, gamma, beta), ...].
        x (jnp.ndarray): Input data for which loss is computed (batch_size, input_dim).
        y (jnp.ndarray): True labels, shape (batch_size,) with values in [0, num_classes).
        l2_lambda (float): Weighting for L2 regularization (default=5e-5).

    Returns:
        total_loss (float): Combined cross-entropy and L2 regularization loss.
    """
    # Forward pass to get logits (unnormalized predictions).
    logits = forward(params, x)

    # Convert labels to one-hot vectors, e.g., label 3 => [0,0,0,1,0,0,0,0,0,0].
    y_one_hot = jax.nn.one_hot(y, num_classes=10)

    # Cross entropy loss:
    # log_softmax(logits) computes log of softmax; multiply with one-hot labels, sum over classes, then mean.
    ce_loss = -jnp.mean(jnp.sum(y_one_hot * jax.nn.log_softmax(logits), axis=-1))

    # L2 weight decay (regularization) to reduce overfitting. 
    # We sum up the squares of all W and multiply by l2_lambda.
    l2_loss = sum(jnp.sum(W**2) for W, _, _, _ in params) * l2_lambda

    # Total loss is cross entropy plus L2 regularization.
    return ce_loss + l2_loss

def compute_accuracy(params, x, y):
    """
    Compute the classification accuracy given parameters, inputs x, and labels y.

    Args:
        params (list): Model parameters.
        x (jnp.ndarray): Input data, shape (batch_size, input_dim).
        y (jnp.ndarray): True labels.

    Returns:
        accuracy (float): The fraction of correct predictions in [0,1].
    """
    # Get logits from the forward pass.
    logits = forward(params, x)

    # Predictions are the class index with the highest probability (softmax).
    predictions = jnp.argmax(jax.nn.softmax(logits), axis=-1)

    # Compare predictions with true labels and take the mean of correct matches.
    return jnp.mean(predictions == y)


# ------------------------- Optimizer Setup ------------------------- #

def create_optimizer(base_lr=0.001, decay_rate=0.98, decay_steps=100):
    """
    Create an Adam optimizer (via Optax) with an exponentially decaying learning rate schedule.

    Args:
        base_lr (float): Initial learning rate.
        decay_rate (float): Multiplicative decay factor.
        decay_steps (int): How often to apply the decay (in steps/batches).

    Returns:
        optimizer (optax.GradientTransformation): The configured Optax optimizer.
    """
    schedule = optax.exponential_decay(
        init_value=base_lr,       # Initial learning rate
        transition_steps=decay_steps,  
        decay_rate=decay_rate
    )
    return optax.adam(schedule)

# Create a global optimizer instance (for convenience).
optimizer = create_optimizer()


# ------------------------- Training Step ------------------------- #

@jit
def update(params, opt_state, x, y):
    """
    Compute the loss and gradient, then update model parameters using the optimizer.
    Includes gradient clipping to prevent excessively large updates.

    Args:
        params (list): Model parameters.
        opt_state (OptState): The current state of the optimizer (e.g., Adam's momentum).
        x (jnp.ndarray): A batch of input data.
        y (jnp.ndarray): Corresponding labels for x.

    Returns:
        params (list): Updated model parameters.
        opt_state (OptState): Updated optimizer state.
        loss_value (float): The computed loss for this batch.
    """
    # Compute both the loss and the gradient w.r.t. parameters.
    loss_value, grads = value_and_grad(cross_entropy_loss)(params, x, y)

    # Clip gradients elementwise between -1.0 and 1.0 to avoid exploding gradients.
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

    # Use Optax to get parameter updates based on the gradients and current optimizer state.
    updates, opt_state = optimizer.update(grads, opt_state, params)

    # Apply the updates to the parameters to get the new parameters.
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_value


# ------------------------- Training Loop ------------------------- #

def train_mlp(train_data, train_labels, test_data, test_labels, num_epochs=200, batch_size=128):
    """
    Train the deep feedforward neural network using the provided dataset.

    Procedure:
    1. Shuffle the dataset each epoch.
    2. Split into mini-batches.
    3. For each batch, call 'update' to perform one gradient descent step.
    4. Periodically compute test loss and accuracy to track performance.

    Args:
        train_data (np.ndarray): Training images of shape (N, 32, 32, 3).
        train_labels (np.ndarray): Training labels of shape (N,).
        test_data (np.ndarray): Test images of shape (M, 32, 32, 3).
        test_labels (np.ndarray): Test labels of shape (M,).
        num_epochs (int): Number of epochs to train (default 200).
        batch_size (int): Mini-batch size (default 128).

    Returns:
        None
    """
    # Start a timer specifically for the training procedure (all epochs).
    train_loop_start = time.time()

    # Number of training examples
    num_train = train_data.shape[0]
    # Number of batches each epoch
    num_batches = num_train // batch_size

    # Create a PRNGKey for parameter initialization
    key = random.PRNGKey(42)

    # Define the sizes of each layer in our network. 
    # For CIFAR-10, the input layer is 32x32x3 = 3072, 
    # and the output layer is 10 for the 10 classes.
    layer_sizes = [3072, 5000, 4000, 3000, 2000, 1000, 10]

    # Initialize all the parameters (weights, biases, gamma, beta) for each layer.
    params = initialize_params(key, layer_sizes)

    # Initialize the optimizer state with the newly created parameters.
    opt_state = optimizer.init(params)

    # Mark the end of "setup" within the training function itself, if desired.
    # (Though some might consider data loading + model init as "setup")
    training_start = time.time()
    setup_time = training_start - train_loop_start
    print(f"Setup time (within train_mlp): {setup_time:.2f} seconds")

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Shuffle the training data and labels in unison.
        perm = np.random.permutation(num_train)
        train_data, train_labels = train_data[perm], train_labels[perm]

        # Process data in mini-batches.
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            x_batch = train_data[start:end].reshape(batch_size, -1)  # Flatten images
            y_batch = train_labels[start:end]

            # Update model parameters given the current batch
            params, opt_state, _ = update(params, opt_state, x_batch, y_batch)

        # Compute test loss and accuracy after each epoch
        test_loss = cross_entropy_loss(params, test_data.reshape(test_data.shape[0], -1), test_labels)
        test_acc = compute_accuracy(params, test_data.reshape(test_data.shape[0], -1), test_labels)

        print(f"Epoch {epoch+1:02d} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")

    # Measure how long the entire training loop took (for all epochs).
    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"Total training (epochs) time: {total_training_time:.2f} seconds")


# ------------------------- Main Execution ------------------------- #

if __name__ == "__main__":
    # Record start time for the entire script
    script_start = time.time()

    # Load the CIFAR-10 dataset (part of setup).
    train_data, train_labels, test_data, test_labels = load_cifar10()

    # Measure the time after data loading (still "setup" overhead).
    data_loading_end = time.time()

    print("\nTraining on a single GPU (or CPU if GPU is not available)...")

    # Call the training function
    train_mlp_start = time.time()
    train_mlp(train_data, train_labels, test_data, test_labels)
    train_mlp_end = time.time()

    # Print separate timing measurements
    print(f"\nTime to load data (script start -> data load end): "
          f"{data_loading_end - script_start:.2f} seconds")

    print(f"Time spent in train_mlp function: "
          f"{train_mlp_end - train_mlp_start:.2f} seconds")

    # Total script end
    script_end = time.time()

    print(f"\nTotal script execution time: {script_end - script_start:.2f} seconds")
