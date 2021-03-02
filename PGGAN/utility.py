import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import autograd


def calc_padding(dims, kernel, stride):
    """
    For constant output: W = (W - F + 2P)/S + 1
    W → Dimensions, F → Kernel, P → Padding, S → Stride
    (S(W - 1) - W + F)/2
    :param dims:
    :param kernel:
    :param stride:
    :return:
    """
    padding = (stride * (dims - 1) - dims + kernel) / 2
    assert padding.is_integer(), "A non-integer result indicates an invalid pairing of dimensions and stride values"
    return int(padding)


def exp_avg_update(target, source, beta):
    """

    :param target:
    :param source:
    :param beta:
    :return:
    """
    with torch.no_grad():
        source_parameters = dict(source.named_parameters())

        for p_name, p_target in target.named_parameters():
            p_source = source_parameters[p_name]
            assert p_source is not p_target
            p_target.copy_(beta * p_target + (1 - beta) * p_source)


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN-GP (https://arxiv.org/abs/1704.00028)
    :param discriminator:
    :param real_samples:
    :param fake_samples:
    :return:
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    alpha = alpha.expand_as(real_samples)  # Broadcast to correct shape
    alpha = alpha.type_as(real_samples)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples.detach() + (1 - alpha) * fake_samples.detach())
    interpolates = interpolates.requires_grad_(True)
    interpolates = interpolates.type_as(real_samples)

    d_interpolates = discriminator(interpolates)
    grad_output = torch.ones_like(d_interpolates)
    grad_output = grad_output.type_as(real_samples)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=grad_output, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(real_samples.shape[0], -1)
    gradient_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def calc_epoch_num(scale_list, batch_sizes, total_images):
    """

    :param scale_list:
    :param batch_sizes:
    :param total_images:
    :return:
    """
    return [np.ceil(scale / total_images) for scale, batch_size in zip(scale_list, batch_sizes)]


class MinibatchSTD(nn.Module):
    def __init__(self):
        """
        Calculates the standard deviation across a batch of feature maps
        """
        super(MinibatchSTD, self).__init__()

    def forward(self, feature_maps, dim=0, epsilon=1e-8):
        # Epsilon → Small value for numerical stability when dividing
        batch, channels, height, width = feature_maps.shape
        std = feature_maps - feature_maps.mean(dim=dim, keepdim=True)  # [B x C x H x W] Subtract mean over batch
        std = torch.sqrt(std.pow(2).mean(dim=dim, keepdim=False) + epsilon)  # [1 x C x H x W]  Calc std over batch
        std = std.mean().view(1, 1, 1, 1)  # Take average over feature_maps and pixels
        std = std.repeat(batch, 1, height, width)  # [B x 1 x H x W]  Replicate over group and pixels
        std = torch.cat([feature_maps, std], 1)  # [B x (C + 1) x H x W]  Append as new feature_map
        return std


class Checkpoint(pl.Callback):
    def __init__(self, filepath, save_rate=100):
        self.filepath = filepath
        self.save_rate = save_rate

    def on_batch_end(self, trainer, pl_module):
        pl_module.batch_count += 1
        if pl_module.batch_count % self.save_rate == 0:
            name = f"{pl_module.name}-Scale({pl_module.scale})-Img({pl_module.img_size})-Iter({pl_module.total_iter}).pth"

            print("Saving Model: ", name)
            torch.save({
                'state_dict': pl_module.state_dict(),
                'settings': pl_module.get_settings()
            }, self.filepath + name)
