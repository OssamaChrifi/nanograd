import numpy as np
import matplotlib.pyplot as plt
from autograd import Tensor
from autograd.modules import Linear, Sequential, ReLU, MSE
from autograd.optim import Adam

# -------------------------------
# 1. Generate a non-linear dataset
# -------------------------------
np.random.seed(42)
# Generate 2000 points between -5 and 5
x = np.linspace(-5, 5, 2000)
# Non-linear function
y = np.sin(x) + 0.5 * np.cos(2 * x)
# Add Gaussian noise (standard deviation 0.1)
noise = np.random.randn(2000) * 0.1
y += noise

# Reshape to have column matrices (shape: [n_samples, 1])
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# -------------------------------
# 2. Convert to Tensors for nanograd
# -------------------------------
x_tensor = Tensor(x, requires_grad=True)
y_tensor = Tensor(y, requires_grad=True)

# -------------------------------
# 3. Create a more complex model
# -------------------------------
# The model is a neural network with two hidden layers and ReLU activation
model = Sequential(
    Linear(1, 16),
    ReLU(),
    Linear(16, 32),
    ReLU(),
    Linear(32, 16),
    ReLU(),
    Linear(16, 1)
)

# -------------------------------
# 4. Define the optimizer and loss function
# -------------------------------
learning_rate = 1e-3
optimizer = Adam(model.parameters, learning_rate)
loss_function = MSE()

# -------------------------------
# 5. Train the model
# -------------------------------
epochs = 10000
loss_history = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_tensor)
    
    # Compute the loss
    loss = loss_function(y_pred, y_tensor)
    
    # Backward pass
    loss.backward()
    
    # Update parameters (SGD step resets gradients)
    optimizer.step()
    
    # Periodically display the loss
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss.data}")
        loss_history.append(loss.data)

print("Training completed.")

# -------------------------------
# 6. Display the results
# -------------------------------
# Plot the real data and the model's prediction
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Real data", color="blue", s=10)
plt.plot(x, y_pred.data, label="Model prediction", color="red", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Non-linear regression with a complex model (nanograd)")
plt.legend()
plt.show()

# Plot the loss evolution
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, epochs, 1000), loss_history, marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss history during training")
plt.show()