import numpy as np
from typing import Tuple, Optional
from .base import Function
from .helper import im2col, col2im

class Add(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data + y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        _, y = self.input
        grad_x = grad_output
        grad_y = grad_output
        shape_y = y.data.shape
        shape_out = grad_output.shape

        if shape_y != shape_out:
            diff = len(shape_out) - len(shape_y)
            new_shape_y = (1,) * diff + shape_y
            axes = tuple(i for i, (dim_out, dim_y) in enumerate(zip(shape_out, new_shape_y)) if dim_y == 1)
            grad_y = grad_output.sum(axis=axes, keepdims=True)
            grad_y = grad_y.reshape(shape_y)

        return grad_x, grad_y
    
class Sub(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data - y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output
    
class Mul(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data * y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output * y.data, grad_output * x.data
    
class Div(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data / y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output / y.data, -grad_output * x.data / (y.data ** 2)
    
class Pow(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data ** y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output * y.data * (x.data ** (y.data - 1)), grad_output * (x.data ** y.data) * np.log(x.data)
    
class MatMul(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data @ y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output @ y.data.T, x.data.T @ grad_output
    
class Sum(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.array([x.data.sum()])
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * np.ones_like(x.data)
    
class Mean(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.array([x.data.mean()])
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * np.ones_like(x.data) / x.data.size

class Exp(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.exp(x.data)
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * np.exp(x.data)

class Log(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.log(x.data)
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output / x.data

class Conv2d(Function):
    def forward(self, x, weight, bias, padding, stride, dilation, groups, padding_mode) -> np.ndarray:
        self.input = (x, weight, bias)
        self.ctx = (x, weight, bias, padding, stride, dilation, groups, padding_mode)
        batch_size, in_channels, _, _ = x.data.shape
        out_channels = weight.data.shape[0]
        if groups == 1:
            x_unf, out_height, out_width = im2col(x.data, weight.data, padding, stride, dilation, padding_mode)
            out_channels = weight.data.shape[0]
            weight_unf = weight.data.reshape(out_channels, -1)
            out_unf = x_unf.transpose(0, 2, 1) @ weight_unf.T + bias.data
            out = out_unf.transpose(0, 2, 1).reshape(x.data.shape[0], out_channels, out_height, out_width)
            self.x_unf = x_unf
            return out
        
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups
        outputs = []

        for g in range(groups):
            x_group = x.data[:, g * in_channels_per_group : (g + 1) * in_channels_per_group, :, :]
            x_unf_group, out_height, out_width = im2col(x_group, weight.data, padding, stride, dilation, padding_mode)
            weight_group = weight.data[g * out_channels_per_group : (g + 1) * out_channels_per_group, :, :, :]
            weight_unf_group = weight_group.reshape(out_channels_per_group, -1)
            bias_group = bias.data[g * out_channels_per_group : (g + 1) * out_channels_per_group]
            out_unf_group = x_unf_group.transpose(0,2,1) @ weight_unf_group.T + bias_group
            out_group = out_unf_group.transpose(0,2,1).reshape(batch_size, out_channels_per_group, out_height, out_width)
            outputs.append(out_group)

        out = np.concatenate(outputs, axis=1)
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, weight, bias, padding, stride, dilation, groups, padding_mode = self.ctx
        batch_size = x.data.shape[0]
        out_channels, in_channels, k_h, k_w = weight.data.shape
        grad_bias = grad_output.sum(axis=(0, 2, 3))
        grad_output_reshaped = grad_output.reshape(batch_size, out_channels, -1)
        grad_weight_unf = np.zeros((out_channels, in_channels * k_h * k_w))
        for i in range(batch_size):
            grad_weight_unf += grad_output_reshaped[i] @ self.x_unf[i].T
        grad_weight = grad_weight_unf.reshape(weight.data.shape)
        grad_x_unf = np.zeros_like(self.x_unf)
        weight_unf = weight.data.reshape(out_channels, -1)
        for i in range(batch_size):
            grad_x_unf[i] = (grad_output_reshaped[i].T @ weight_unf).T
        grad_x = col2im(grad_x_unf, x.data.shape, (k_h, k_w), padding, stride, dilation)
        return grad_x, grad_weight, grad_bias
