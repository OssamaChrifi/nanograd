import numpy as np

PAD_MODES = {
            'zeros': ('constant', {'constant_values': 0}),
            'reflect': ('reflect', {}),
            'replicate': ('edge', {}),
            'circular': ('wrap', {})
        }

def im2col(x, weight, padding, stride, dilation, padding_mode):
        batch_size = x.shape[0]
        mode, kwargs = PAD_MODES[padding_mode]
        _, _, kernel_size_h, kernel_size_w = weight.shape
        x_padded = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode=mode, **kwargs)
        kernel_effective_h = dilation * (kernel_size_h - 1) + 1
        kernel_effective_w = dilation * (kernel_size_w - 1) + 1
        out_height = (x_padded.shape[2] - kernel_effective_h) // stride + 1
        out_width = (x_padded.shape[3] - kernel_effective_w) // stride + 1
        windows = np.lib.stride_tricks.sliding_window_view(x_padded, (kernel_effective_h, kernel_effective_w), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :]
        windows = windows[:, :, :, :, ::dilation, ::dilation]
        col = windows.transpose(0, 2, 3, 1, 4, 5)  
        col = col.reshape(batch_size, out_height * out_width, -1).transpose(0, 2, 1)  
        return col, out_height, out_width
    
def col2im(cols, x_shape, kernel_size, padding, stride, dilation):
    batch_size, channels, height, width = x_shape
    k_h, k_w = kernel_size
    out_h = (height + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    cols_reshaped = cols.reshape(batch_size, channels, k_h, k_w, out_h, out_w)
    x_padded = np.zeros((batch_size, channels, height + 2 * padding, width + 2 * padding))
    for i in range(k_h):
        for j in range(k_w):
            x_padded[:, :, i * stride : i * stride + out_h, j * stride : j * stride + out_w] += cols_reshaped[:, :, i, j, :, :]
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded