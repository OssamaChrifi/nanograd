import numpy as np
import matplotlib.pyplot as plt
from autograd import Tensor
from autograd.modules import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, CrossEntropyLoss
from autograd.optim import Adam

# -------------------------------
# Load the MNIST dataset (using Keras for simplicity)
# -------------------------------
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images and add a channel dimension (required by Conv2D)
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(-1, 1, 28, 28)  # shape: (batch, channels, height, width)
x_test  = x_test.reshape(-1, 1, 28, 28)

# -------------------------------
# 1. Set training parameters and batch size
# -------------------------------
batch_size = 64
n_samples = x_train.shape[0]
n_batches = n_samples // batch_size  # Use integer division for full batches
epochs = 6
learning_rate = 1e-3

# -------------------------------
# 2. Define the model (same as before)
# -------------------------------
model = Sequential(
    Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    Linear(16 * 7 * 7, 10)  # 10 output classes for MNIST
)

optimizer = Adam(model.parameters, learning_rate)
loss_function = CrossEntropyLoss()  # Assumes logits + integer labels

loss_history = []

# -------------------------------
# 3. Training loop with mini-batches
# -------------------------------
for epoch in range(epochs):
    epoch_loss = 0
    # Shuffle training data at the beginning of each epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for i in range(n_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        # Create batch tensors; we create new Tensors each time to allow gradient tracking
        batch_x = Tensor(x_train[batch_indices], requires_grad=True)
        batch_y = Tensor(y_train[batch_indices], requires_grad=False)
        
        # Forward pass
        logits = model(batch_x)
        loss = loss_function(logits, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.data
    
    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch} - Avg Loss: {avg_loss}")

print("Training completed.")

# -------------------------------
# 4. Evaluate the model on the test set
# -------------------------------
# Wrap the whole test set in a Tensor (here, we assume test set is small enough)
x_test_tensor = Tensor(x_test, requires_grad=False)
logits_test = model(x_test_tensor)
predictions = np.argmax(logits_test.data, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# 5. Plot training loss
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()

# -------------------------------
# 6. Display some test predictions
# -------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(x_test[i, 0], cmap='gray')
    ax.set_title(f"GT: {y_test[i]}\nPred: {predictions[i]}")
    ax.axis('off')
plt.suptitle("Sample Predictions on MNIST Test Set")
plt.show()
