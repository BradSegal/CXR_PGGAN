import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from GAN.utility import calc_padding


def he_initializer(module):
    """
    Returns He's Initialization Constant for Conv2D or linear modules. It is inversely proportional to the root
    of the product of the neurons/weights for a given module. Scales the gradient relative to the number of weights
    to remove the correlation between the number of connections and the gradient.
    Formulation only valid for convolutional & linear layers due to weight arrangement
    https://arxiv.org/abs/1502.01852
    """
    assert isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)), \
        "Formulation only valid for Conv2d & linear layers"
    weight_shape = module.weight.size()  # due to arrangement of weights
    # Conv2d weights as out_channel x in_channel x kernel[0] x kernel[1]

    fan_in = np.prod(weight_shape[1:])  # Removes the out_channel weights and multiplies the rest together
    he_const = np.sqrt(2.0 / fan_in)

    return he_const


class PixelNormalizationLayer(nn.Module):
    def __init__(self):
        """
        Normalizes a minibatch of images by dividing each pixel by the average squared pixel across all channels
        Norm = Root(Pixel / Sum(Pixel**2)/(Num Channels))
        """
        super(PixelNormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        # Epsilon → Small value for numerical stability when dividing
        norm = x * (x.pow(2).mean(dim=1, keepdim=True) + epsilon).rsqrt()  # rsqrt → Reciprocal Square Root
        return norm


class EqualizedLayer(nn.Module):
    def __init__(self, module, equalize=True, bias_init=True, lrmult=1.0):
        """
        Wrapper layer that enables a linear or convolutional layer to execute He Initialization at runtime as well
        as set initial biases of a module to 0.
        The initialization is performed during the forward pass of the network to enable adaptive gradient descent methods
        (eg. Adam) to better compensate for the equalization of learning rates. Equalization first sets all weights to random
        numbers between -1 & 1 / N(0, 1), and then multiplies by the He constant at runtime.
        :param module: Torch module to be equalized based on the number of connections
        :param equalize: Flag to disable He Initialization
        :param bias_init: Flag to disable initializing bias values to 0
        :param lrmult: Custom layer-specific learning rate multiplier
        """
        super(EqualizedLayer, self).__init__()

        self.module = module
        self.equalize = equalize
        self.init_bias = bias_init

        if self.equalize:
            self.module.weight.data.normal_(0, 1)  # Normal distribution mean of 0, SD of 1
            self.module.weight.data /= lrmult  # Scale weights by a layer specific learning rate multiplier
            # Divides by multiplier as the He Value is the reciprocal of multiple of the output weights
            self.he_val = he_initializer(self.module)
        if self.init_bias:
            self.module.bias.data.fill_(0)

    def forward(self, x):
        x = self.module(x)  # Forward pass through the module
        if self.equalize:
            x *= self.he_val  # Scale by the He Constant
        return x


class EqualizedConv2D(EqualizedLayer):
    def __init__(self, prev_channels, channels, kernel=3, stride=1, padding=0, bias=True, transpose=False, **kwargs):
        """
        Modified 2D convolution that is able to employ He Initialization at runtime as well as to initialize biases to 0
        :param prev_channels:
        :param channels:
        :param kernel:
        :param stride:
        :param padding:
        :param bias:
        :param transpose:
        :param kwargs:
        """
        if not transpose:
            conv = nn.Conv2d(in_channels=prev_channels,
                             out_channels=channels,
                             kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             bias=bias)
        else:
            conv = nn.ConvTranspose2d(in_channels=prev_channels,
                                      out_channels=channels,
                                      kernel_size=kernel,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias)

        EqualizedLayer.__init__(self, conv, **kwargs)


class EqualizedLinear(EqualizedLayer):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        """
        Modified Fully Connected Layer to employ He Initialization at runtime and initialize biases to 0
        :param in_features:
        :param out_features:
        :param bias:
        :param kwargs:
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        linear = nn.Linear(in_features, out_features, bias=bias)
        EqualizedLayer.__init__(self, linear, **kwargs)


def Conv_Block(prev_channel, channels, kernel, stride, padding, bias=True,
                   equalize=True, leakiness=0.2, normalize=True, activation=True, transpose=False):
    """
    Convenience method to simplify the logic of creating a convolution block by specifying the layers to return.
    Allows specification of equalized learning rate convolutions and normalization layers after convolutions.
    :param prev_channel:
    :param channels:
    :param kernel:
    :param stride:
    :param padding:
    :param bias:
    :param equalize:
    :param leakiness:
    :param normalize:
    :param activation:
    :param transpose:
    :return:
    """
    block = nn.ModuleList()
    if equalize:
        block.append(EqualizedConv2D(prev_channels=prev_channel, channels=channels, kernel=kernel,
                                     stride=stride, padding=padding, bias=bias, transpose=transpose))
    else:
        block.append(nn.Conv2d(in_channels=prev_channel, out_channels=channels, kernel_size=kernel,
                               stride=stride, padding=padding, bias=bias))
    if activation:
        block.append(nn.LeakyReLU(negative_slope=leakiness, inplace=True))

    if normalize:
        block.append(PixelNormalizationLayer())

    return block


class ScaleBlock(nn.Module):
    def __init__(self, dims, prev_channel, channels, scale=1, equalize=True, normalize=True,
                 leakiness=0.2, kernel=3, stride=1, padding=None, bias=True, mode="bilinear"):
        """
        Standard convolutional block that combines two identical convolutions and an interpolation operation.
        If the block upscales an image, the upscaling is done prior to the convolutions
        If the block downscales, the upscaling is done after the convolutions
        :param dims:
        :param prev_channel:
        :param channels:
        :param scale:
        :param equalize:
        :param normalize:
        :param leakiness:
        :param kernel:
        :param stride:
        :param padding:
        :param bias:
        :param mode:
        """
        super(ScaleBlock, self).__init__()

        assert scale in [0.5, 1, 2], "Scale can only half, double or maintain spatial resolution"
        self.scale = scale
        self.equalize = equalize

        assert mode in ['nearest', 'bilinear'], f"Only configured for 'nearest' & 'bilinear', but {mode} was selected"
        self.mode = mode

        if padding is None:
            padding = calc_padding(dims, kernel, stride)

        self.convolv = nn.Sequential(
            *Conv_Block(prev_channel=prev_channel, channels=channels, kernel=kernel, stride=stride, padding=padding,
                            bias=bias, equalize=equalize, leakiness=leakiness, normalize=normalize),
            *Conv_Block(prev_channel=channels, channels=channels, kernel=kernel, stride=stride, padding=padding,
                            bias=bias, equalize=equalize, leakiness=leakiness, normalize=normalize)
        )

    def forward(self, feat_map):
        if self.scale > 1:
            feat_map = F.interpolate(input=feat_map, scale_factor=self.scale, mode=self.mode)

        feat_map = self.convolv(feat_map)

        if self.scale < 1:
            feat_map = F.avg_pool2d(input=feat_map, kernel_size=(2, 2))

        return feat_map
