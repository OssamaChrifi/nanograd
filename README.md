# nanograd

**nanograd** is a lightweight deep learning framework built from scratch. It implements a custom automatic differentiation engine (autograd) and provides a set of neural network modules (such as linear layers, convolutional layers, pooling layers, and activation functions) along with optimizers like SGD and Adam. The framework is designed for educational purposes, allowing you to understand the inner workings of deep learning libraries while producing performance comparable to PyTorch.

## Features

- **Automatic Differentiation:**  
  Compute gradients automatically via backpropagation using a custom autograd engine.

- **Modular Neural Network Components:**  
  - **Layers:** Linear, Conv2d, MaxPool2d, Flatten  
  - **Activations:** ReLU, Sigmoid, Tanh, Softmax  
  - **Losses:** MSE, MAE, CrossEntropyLoss  
  - **Optimizers:** SGD, Adam

- **Ease of Extension:**  
  A minimalistic design that makes it easy to add new modules or experiment with custom operations.

- **Performance Comparability:**  
  Tests show that nanograd produces outputs and gradients similar to those of PyTorch.

## Usage

nanograd comes with several example scripts that demonstrate how to build and train models.

**Regression Example**
A regression test script demonstrates training a neural network on a non-linear function. The model consists of multiple layers with ReLU activations, and the script plots the model’s predictions alongside the training loss.

```python
python -m 'test.linear_regression'
```

**Classification Example**
The classification test uses the MNIST dataset to build and train a convolutional neural network. It includes Conv2d, MaxPool2d, and a fully connected layer, and processes the data in mini-batches.

```python
python -m 'test.MNIST_classification'
```
## TODO / Future Updates
Extend Module Library:
Add more layers and modules (e.g., Batch Normalization, Dropout, RNNs, etc.).

Advanced Optimizers:
Implement additional optimizers (e.g., RMSProp, Adagrad) and learning rate schedulers.

Improved Documentation:
Expand the documentation with more examples, tutorials, and API references.

Performance Optimization:
Optimize the autograd engine and memory management for faster computations and reduced memory footprint.

GPU Support:
Explore ways to integrate GPU acceleration for the framework.

Integration Tests:
Develop a comprehensive test suite with integration tests to ensure robustness and correctness across updates.


## Acknowledgements
nanograd was built as an educational tool to deepen the understanding of automatic differentiation and neural network training. Special thanks to the developers and communities behind PyTorch and other deep learning frameworks for their inspiring work.