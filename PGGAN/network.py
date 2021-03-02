import copy
from collections import OrderedDict  # Additional Data Type for Naming Elements of Networks

import numpy as np  # General Mathematics and Linear Algebra Library
import pytorch_lightning as pl
import torch  # Overarching framework for PyTorch
import torch.nn as nn  # General Neural Network Building Blocks
import torch.nn.functional as F  # Neural Network Functions
import torchvision.utils as vutils  # Additional Image Utilities

from GAN.custom_layers import EqualizedLinear, Conv_Block, ScaleBlock, PixelNormalizationLayer
from GAN.dataset import VariableImageFolder
from GAN.utility import MinibatchSTD, compute_gradient_penalty, exp_avg_update, calc_padding


class ProGenerator(nn.Module):
    def __init__(self, z_dim=512, channel_depth=512, init_bias=True, norm_layers=True,
                 out_channels=3, equalize_layers=True, leakiness=0.2, mode="nearest"):
        super(ProGenerator, self).__init__()
        # Store Model Parameters
        self.z_dim = z_dim  # Dimensions of latent space vector

        self.channel_depth = [channel_depth]  # Initial Number of channels to produce from latent space
        # Model begins by producing 4x4 images, incremented when alpha reaches 1
        self.register_buffer('current_size', torch.tensor(4))
        # Current number of completed blocks, incremented when alpha reaches 1 for a given layer
        self.register_buffer('current_depth', torch.tensor(0))
        self.register_buffer('alpha', torch.tensor(0))  # Mixing co-efficient for use when upscaling the network

        self.init_bias = init_bias  # Initialize bias to 0
        self.norm_layers = norm_layers  # Whether to apply minibatch normalization layer
        self.out_channels = out_channels  # The final number of colour channels used in the generated image
        self.equalize = equalize_layers  # Whether to use the He Constant to equalize layer outputs at runtime
        self.leakiness = leakiness  # The co-efficient of the negative slope of the Leaky ReLU activation
        self.mode = mode  # Interpolation mode for upscaling, Paper utilizes nearest neighbour mode

        # Define Layer Architectures
        self.latent_linear = nn.Sequential(EqualizedLinear(in_features=z_dim, out_features=16 * channel_depth,
                                                           equalize=equalize_layers, bias_init=init_bias),
                                           # Initial latent space processing
                                           nn.LeakyReLU(negative_slope=leakiness, inplace=True))

        self.init_conv = nn.Sequential(  # Initial convolution on latent space after initial linear processing
            *Conv_Block(prev_channel=channel_depth, channels=channel_depth,
                            # Convolutions maintain number of channels
                            kernel=3, stride=1, padding=calc_padding(dims=4, kernel=3, stride=1), bias=True,
                            equalize=equalize_layers, leakiness=leakiness, normalize=norm_layers))

        self.ScaleBlocks = nn.ModuleList()  # Stores list of scaling blocks to double spatial resolutions
        # Stores the feature map to RGB convolutions A new one is needed for each network expansion
        self.toRGB = nn.ModuleList()
        # The RGB layers are stored to enable extracting smaller intermediary images from scaling blocks
        self.toRGB.append(*Conv_Block(prev_channel=channel_depth, channels=out_channels,
                                          kernel=1, stride=1, padding=0, bias=True, equalize=equalize_layers,
                                          activation=False,
                                          normalize=False))  # The final convolution acts as an activation function

        if self.norm_layers:
            self.norm = PixelNormalizationLayer()

    def forward(self, x):
        if self.norm_layers:
            x = self.norm(x)

        features = np.prod(x.size()[1:])  # Multiple of all dimensions except batch dimension = Total Feature Number
        x = x.view(-1, features)  # Batch dimension x Features

        batch_size = x.size()[0]
        x = self.latent_linear(x)  # Initial Latent Processing & Formatting
        x = x.view(batch_size, -1, 4, 4)  # Reshape to Batch x Depth x 4 x 4

        x = self.init_conv(x)  # Perform initial 3x3 convolution without upscaling

        if self.alpha > 0 and self.current_depth == 1:  # Apply mixing for when the network begins to expand
            # Expansion determined when alpha is incremented with the completed depth layers still at 0
            expansion = self.toRGB[-2](x)
            expansion = F.interpolate(input=expansion, scale_factor=2, mode=self.mode)

        for scale_num, scale_block in enumerate(self.ScaleBlocks, 1):
            # Start at 1 due to the first image dimension not requiring scaling
            x = scale_block(x)  # Process the input through the expansion block of upscale, conv, conv

            if self.alpha > 0 and (scale_num == self.current_depth - 1):
                expansion = self.toRGB[-2](x)
                expansion = F.interpolate(input=expansion, scale_factor=2, mode=self.mode)

        x = self.toRGB[-1](x)  # Final layer to RGB

        if self.alpha > 0:
            x = self.alpha * expansion + (1.0 - self.alpha) * x  # Mix the inputs at the final scale

        return x

    def incrementdepth(self, new_depth):
        """
        Adds scaling block to the model, doubles the spatial resolution of the final image

        """
        device = next(self.parameters()).device
        self.current_depth += 1
        self.current_size *= 2

        prev_depth = self.channel_depth[-1]
        self.channel_depth.append(new_depth)
        # Adds scaling block, padding is calculated from the spatial dimensions and filter properties
        size = self.current_size.cpu().numpy()
        self.ScaleBlocks.append(ScaleBlock(dims=size, prev_channel=prev_depth,
                                           channels=new_depth, scale=2, equalize=self.equalize,
                                           normalize=self.norm_layers, leakiness=self.leakiness, kernel=3,
                                           stride=1, padding=None, bias=True, mode=self.mode).to(device))

        self.toRGB.append(*Conv_Block(prev_channel=new_depth, channels=self.out_channels,
                                      kernel=1, stride=1, padding=0, bias=True, equalize=self.equalize,
                                      activation=False, normalize=False).to(device))

    def set_alpha(self, new_alpha):
        """
        Sets the mixing factor used when upscaling the network. Alters the functioning of the forward function
        to include the second last layer and interpolate between it and the final output of the added scaling block.
        """
        if new_alpha < 0 or new_alpha > 1:
            raise ValueError("Alpha must be in the range [0,1]")

        self.alpha = new_alpha

    def load(self, checkpoint):
        """
        Automatically scales the network to the required size and loads the weights
        :param checkpoint: Saved network state
        :return:
        """
        for depth in checkpoint['settings']['channel_depth'][1:]:
            self.incrementdepth(depth)
        self.load_state_dict(checkpoint['state_dict'])
        print("Generator Weights Loaded")


class ProDiscriminator(nn.Module):
    def __init__(self, channel_depth=512, init_bias=True, norm_layers=False, input_channels=3,
                 decision_layer_dim=1, equalize_layers=True, leakiness=0.2, minibatch_std=True):
        super(ProDiscriminator, self).__init__()
        # Store Model Parameters
        self.input_channels = input_channels
        self.decision_layer_dim = decision_layer_dim  # Can be augmented to allow for classification of an image

        self.channel_depth = [channel_depth]  # Initial Number of channels to produce from image
        self.register_buffer('current_size', torch.tensor(4))  # Current size of images for descrimination
        self.register_buffer('current_depth', torch.tensor(
            0))  # Current number of completed blocks, incremented when the generator completes mixing
        self.register_buffer('alpha', torch.tensor(0))  # Mixing co-efficient for use when upscaling the network

        self.init_bias = init_bias  # Initialize bias to 0
        self.norm_layers = norm_layers  # Whether to apply minibatch normalization layer
        self.minibatch = minibatch_std  # Whether to calculate the std of all layers prior to the final convolution
        self.equalize = equalize_layers  # Whether to use the He Constant to equalize layer outputs at runtime
        self.leakiness = leakiness  # The co-efficient of the negative slope of the Leaky ReLU activation

        if self.minibatch:
            self.miniSTD = MinibatchSTD()

        self.fromRGB = nn.ModuleList()

        self.fromRGB.append(*Conv_Block(prev_channel=input_channels, channels=channel_depth,
                                            kernel=1, stride=1, padding=0, bias=True, equalize=equalize_layers,
                                            activation=False, normalize=False))  # Initial RGB to Feature Map Processing

        self.ScaleBlocks = nn.ModuleList()  # Will be added to as the network grows
        # Blocks will be added to the front of the list as the descriminator grows at its input end

        self.DecisionBlock = nn.Sequential(
            *Conv_Block(prev_channel=channel_depth + minibatch_std, channels=channel_depth,  # Add channel for std
                        kernel=3, stride=1, padding=calc_padding(4, 3, 1), bias=True, equalize=equalize_layers,
                        activation=True, normalize=norm_layers),  # eg: 513x4x4 → 512x4x4
            *Conv_Block(prev_channel=channel_depth, channels=channel_depth,
                        kernel=4, stride=1, padding=0, bias=True, equalize=equalize_layers,
                        activation=True, normalize=norm_layers))  # eg: 512x4x4 → 512x1x1

        self.DecisionLinear = EqualizedLinear(in_features=channel_depth, out_features=decision_layer_dim,
                                              equalize=self.equalize, bias_init=self.init_bias)  # eg: 512x1x1 → 1x1x1

    def forward(self, x):

        if self.alpha > 0 and len(
                self.fromRGB) > 1:  # Check if the layers are mixed and multiple resolutions are being used
            pooled = F.avg_pool2d(input=x,
                                  kernel_size=(2, 2))  # Downsample the larger image to mix with the output of the
            pooled = self.fromRGB[-2](pooled)  # new processing layer

        # Convert from RGB to Feature Maps:
        x = self.fromRGB[-1](x)

        for scale_num, scale_block in enumerate(self.ScaleBlocks, 1):
            # Start at 1 due to the first image dimension not requiring scaling
            x = scale_block(x)  # Process the input through the contraction block of conv, conv, downsample

            # Processing blocks are added to the front of the array, so mixing always occurs at the top layer
            if self.alpha > 0 and scale_num == 1:
                x = self.alpha * pooled + (1.0 - self.alpha) * x  # Interpolate the downsampled input and the new input

        if self.minibatch:
            x = self.miniSTD(x)

        x = self.DecisionBlock(x)  # Final 3x3, 4x4 conv

        features = np.prod(x.size()[1:])
        x = x.view(-1, features)

        x = self.DecisionLinear(x)  # Returns logits

        return x  # Final score for whether the image is fake

    def incrementdepth(self, new_depth):
        """
        Adds scaling block to the model, doubles the spatial resolution of the inputted image
        Includes two channel depth parameters as the descriminator often has different channels for each convolution
        """
        device = next(self.parameters()).device
        self.current_depth += 1
        self.current_size *= 2

        prev_depth = self.channel_depth[-1]
        self.channel_depth.append(new_depth)
        size = self.current_size.cpu().numpy()

        self.ScaleBlocks.insert(0, ScaleBlock(dims=size, prev_channel=new_depth,  # Add block to the beginning of list
                                              channels=prev_depth, scale=0.5, equalize=self.equalize,
                                              normalize=self.norm_layers, leakiness=self.leakiness, kernel=3,
                                              stride=1, padding=None, bias=True).to(device))

        self.fromRGB.append(*Conv_Block(prev_channel=self.input_channels, channels=new_depth,
                                            kernel=1, stride=1, padding=0, bias=True, equalize=self.equalize,
                                            activation=False, normalize=False).to(device))

    def set_alpha(self, new_alpha):
        """
        Sets the mixing factor used when incrementing the network. Alters the functioning of the forward function
        to include the second layer and interpolate between it and the initial output of the added scaling block.
        """
        if new_alpha < 0 or new_alpha > 1:
            raise ValueError("Alpha must be in the range [0,1]")
        self.alpha = new_alpha

    def save(self, filename):
        settings = {
            'input_channels': self.input_channels,
            'decision_layer_dim': self.decision_layer_dim,
            'channel_depth': self.channel_depth,
            'init_bias': self.init_bias,
            'norm_layers': self.norm_layers,
            'minibatch': self.minibatch,
            'equalize': self.equalize,
            'leakiness': self.leakiness
        }
        torch.save({'state_dict': self.state_dict(),
                    'settings': settings}, filename)

    def load(self, checkpoint):
        """
        Automatically scales the network to the required size and loads the weights
        :param checkpoint: Saved network state
        :return:
        """
        for depth in checkpoint['settings']['channel_depth'][1:]:
            self.incrementdepth(depth)
        self.load_state_dict(checkpoint['state_dict'])
        print("Discriminator Weights Loaded")


class ProGAN(pl.LightningModule):
    def __init__(self, hparams):
        super(ProGAN, self).__init__()
        self.hparams = hparams

        self.name = hparams.name
        self.directory = hparams.directory
        self.batch_sizes = [int(batch) for batch in hparams.batch_sizes]

        self.worker_num = hparams.worker_num
        self.register_buffer('img_size', torch.tensor(4))  # initial spatial dimensions
        self.z_dim = hparams.z_dim

        self.generator = ProGenerator(z_dim=hparams.z_dim, channel_depth=hparams.initial_channel_depth,
                                      init_bias=hparams.init_bias, norm_layers=hparams.g_norm_layers,
                                      out_channels=hparams.img_channels, equalize_layers=hparams.equalize,
                                      leakiness=hparams.leakiness, mode=hparams.mode)

        self.discriminator = ProDiscriminator(channel_depth=hparams.initial_channel_depth, init_bias=hparams.init_bias,
                                              norm_layers=hparams.d_norm_layers, input_channels=hparams.img_channels,
                                              decision_layer_dim=hparams.decision_dim, equalize_layers=hparams.equalize,
                                              leakiness=hparams.leakiness, minibatch_std=hparams.minibatch_std)

        self.register_buffer('scale', torch.tensor(0))
        self.lr = hparams.lr
        self.b1 = hparams.b1
        self.b2 = hparams.b2
        self.eps = hparams.eps
        self.drift = hparams.drift
        self.lambda_val = hparams.lambda_val

        self.imgs = None

        self.channel_depths = [int(depths) for depths in hparams.depths]

        self.register_buffer('alpha', torch.tensor(0))
        self.register_buffer('alpha_jumps', hparams.alpha_jumps)
        iters = int(self.alpha_jumps[self.scale] / self.batch_sizes[self.scale])
        self.alpha_iters = iters
        self.alpha_vals = np.linspace(start=1, stop=0, num=iters)

        self.sample_rate = hparams.sample
        self.sample_num = hparams.sample_num

        self.iteration_sets = [int(iteration) for iteration in hparams.iterations]
        self.register_buffer('total_iter', torch.tensor(0))
        self.register_buffer('current_iter', torch.tensor(0))
        self.register_buffer('num_images_seen', torch.tensor(0))
        self.register_buffer('iter_images_seen', torch.tensor(0))
        self.register_buffer('g_current_iter', torch.tensor(0))
        self.register_buffer('d_current_iter', torch.tensor(0))

        self.register_buffer('fixed_z', torch.randn(self.sample_num, self.z_dim, requires_grad=False))

        self.register_buffer('epoch_counter', torch.tensor(0))
        self.register_buffer('batch_count', torch.tensor(0))

        self.ema = hparams.ema
        if self.ema:
            self.ema_decay = hparams.ema_decay
            self.gen_shadow = copy.deepcopy(self.generator)
            exp_avg_update(target=self.gen_shadow, source=self.generator, beta=0)

        self.set_fp16 = hparams.set_fp16

    def forward(self, latent):
        return self.generator(latent)

    def training_step(self, img_list, batch_idx, optimizer_idx):
        device = next(self.parameters()).device
        img_batch, _ = img_list

        self.check_network_scale()

        # Random Noise Vector:
        latent = torch.randn(img_batch.shape[0], self.z_dim)
        latent = latent.type_as(img_batch)
        # Training the discriminator first appears to promote more stable training
        if optimizer_idx == 0:  # Discriminator Training:
            # No labels are needed as the higher the average real score the better the discriminator
            # and the lower the average fake score the better
            real_imgs = img_batch.detach()
            real_validity = self.discriminator(real_imgs)

            fake_imgs = self.forward(latent).detach()
            fake_validity = self.discriminator(fake_imgs)

            gradient_penalty = compute_gradient_penalty(real_samples=real_imgs, fake_samples=fake_imgs)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_val * gradient_penalty \
                     + self.drift * torch.mean(real_validity ** 2)
            d_loss = d_loss.unsqueeze(0)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'opt_step': torch.tensor([optimizer_idx]).to(device)
            })

            return output

        if optimizer_idx == 1:  # Generator Training:
            fake_imgs = self.forward(latent)  # Generate Images from Noise
            validity_score = self.discriminator(fake_imgs)

            # Calculate Generator Loss
            g_loss = -torch.mean(validity_score)
            g_loss = g_loss.unsqueeze(0)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'opt_step': torch.tensor([optimizer_idx]).to(device)
            })

            return output

    def training_step_end(self, outputs):
        """
        Method collating all results on a single GPU. Can be used to increment values that will be distributed
        to all GPUs on future training steps. Image logs can be produced without redundancy as well.
        """
        if int(outputs['opt_step'][-1]) == 1:  # Only called after the generator is updated
            device = next(self.parameters()).device

            self.total_iter += 1
            self.current_iter += 1
            self.num_images_seen += self.batch_sizes[self.scale]
            self.iter_images_seen += self.batch_sizes[self.scale]
            self.g_current_iter += 1

            self.check_network_scale()

            if self.ema:
                exp_avg_update(target=self.gen_shadow, source=self.generator, beta=self.ema_decay)

            if self.alpha_iters >= self.current_iter:
                if len(self.alpha_vals) > 0:  # Check alpha values exist for current scale
                    alpha = self.alpha_vals[self.current_iter - 1]
                    self.update_alpha(torch.tensor(alpha).to(device))
                else:  # Correct if no mixing is specified for particular scale
                    self.update_alpha(torch.tensor(0).to(device))
            else:
                if self.alpha > 0:  # Correct if alpha not 0
                    self.update_alpha(torch.tensor(0).to(device))

            self.logger.experiment.add_scalar('Alpha_Mixing_Coefficient', self.alpha, self.total_iter)

            # Generate Fixed Samples & Store
            if self.current_iter % self.sample_rate == 0:
                with torch.no_grad():
                    samples = self.forward(self.fixed_z).detach()
                    grid = vutils.make_grid(samples, normalize=True)
                    self.logger.experiment.add_image('generated_images', grid, self.total_iter)

                    if self.ema:
                        samples = self.gen_shadow.forward(self.fixed_z).detach()
                        grid = vutils.make_grid(samples, normalize=True)
                        self.logger.experiment.add_image('shadow_generated_images', grid, self.total_iter)
        else:
            self.d_current_iter += 1

        outputs['loss'] = outputs['loss'].squeeze(0)
        del outputs['opt_step']  # Remove tracking value from output
        return outputs

    def check_network_scale(self):
        device = next(self.parameters()).device
        if self.generator.current_depth != self.scale:
            self.generator.current_depth = torch.tensor(self.scale).to(device)
            if self.ema:
                self.gen_shadow.current_depth = torch.tensor(self.scale).to(device)

        if self.discriminator.current_depth != self.scale:
            self.discriminator.current_depth = torch.tensor(self.scale).to(device)

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        epsilon = self.eps

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2), eps=epsilon)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2), eps=epsilon)

        return [opt_d, opt_g], []

    def configure_apex(self, amp, model, optimizers, amp_level):
        [model.generator, model.discriminator], optimizers = amp.initialize(
            [model.generator, model.discriminator], optimizers, opt_level=amp_level,
        )

        return model, optimizers

    def add_depth(self):
        device = next(self.parameters()).device
        scale = self.scale

        print("Adding Depth, Scale:", scale + 1, "Img Size:", self.img_size * 2)

        self.generator.incrementdepth(self.channel_depths[scale])

        if self.ema:
            self.gen_shadow.incrementdepth(self.channel_depths[scale])

        self.discriminator.incrementdepth(self.channel_depths[scale])

        self.img_size *= 2  # Double Spatial Dimensions
        self.scale += 1
        self.current_iter = torch.tensor(0).to(device)
        self.iter_images_seen = torch.tensor(0).to(device)
        # Generate a smooth linear array of numbers from 1 to 0 for fading in layers
        iters = int(self.alpha_jumps[self.scale] / self.batch_sizes[self.scale])
        self.alpha_iters = iters
        alpha_range = np.linspace(start=1, stop=0, num=iters)
        self.alpha_vals = alpha_range
        self.update_alpha(torch.tensor(1).to(device))

    def update_alpha(self, new_alpha):
        if type(new_alpha).__name__ != 'Tensor':
            device = next(self.parameters()).device
            new_alpha = torch.tensor(new_alpha).to(device)

        self.alpha = new_alpha
        self.generator.set_alpha(new_alpha)
        if self.ema:
            self.gen_shadow.set_alpha(new_alpha)
        self.discriminator.set_alpha(new_alpha)

    def prepare_data(self):
        self.imgs = VariableImageFolder(self.directory)

    def train_dataloader(self):  # Dataloader is reset at the end of each epoch
        # Using the variable length dataset enables finer control over the number of images shown
        batch_size = self.batch_sizes[self.scale]
        img_size = self.img_size
        worker_num = self.worker_num
        imgs_needed = self.iteration_sets[self.scale] - self.iter_images_seen  # For loading in the middle of epochs
        if imgs_needed <= 0:
            imgs_needed = 1  # Correct for potential image number error

        self.imgs.update_img_size(img_size)
        self.imgs.update_img_len(imgs_needed)

        dataloader = torch.utils.data.DataLoader(self.imgs, batch_size=batch_size, drop_last=True,
                                                 shuffle=True, num_workers=worker_num)
        return dataloader

    def on_epoch_end(self):
        self.epoch_counter += 1
        # Number of total images needed
        imgs_needed = self.iteration_sets[self.scale] - self.iter_images_seen
        if imgs_needed <= 0:
            self.add_depth()

    def get_settings(self):
        settings_dict = {
            'alpha': self.alpha,

            'g_size': self.generator.current_size,
            'g_depth': self.generator.current_depth,
            'g_alpha': self.generator.alpha,
            'd_size': self.discriminator.current_size,
            'd_depth': self.discriminator.current_depth,
            'd_alpha': self.discriminator.alpha,

            'total_iter': self.total_iter,
            'current_iter': self.current_iter,
            'num_images_seen': self.num_images_seen,
            'iter_images_seen': self.iter_images_seen,
            'g_current_iter': self.g_current_iter,
            'd_current_iter': self.d_current_iter,

            'fixed_z': self.fixed_z
        }
        return settings_dict

    def set_settings(self, settings):

        self.alpha = settings['alpha']
        self.update_alpha(settings['alpha'])  # Update alpha to previous value
        iters = int(self.alpha_jumps[self.scale] / self.batch_sizes[self.scale])
        self.alpha_iters = iters
        self.alpha_vals = np.linspace(start=1, stop=0, num=iters)

        self.generator.current_size = settings['g_size']
        self.generator.current_depth = settings['g_depth']

        if self.ema:
            self.gen_shadow.current_size = settings['g_size']
            self.gen_shadow.current_depth = settings['g_depth']

        self.discriminator.current_size = settings['d_size']
        self.discriminator.current_depth = settings['d_depth']

        self.total_iter = settings['total_iter']
        self.current_iter = settings['current_iter']
        self.num_images_seen = settings['num_images_seen']
        self.iter_images_seen = settings['iter_images_seen']
        self.g_current_iter = settings['g_current_iter']
        self.d_current_iter = settings['d_current_iter']

        self.fixed_z = settings['fixed_z']

    def load_model(self, path):
        checkpoint = torch.load(path)
        for _ in range(checkpoint['settings']['g_depth']):
            self.add_depth()
        self.load_state_dict(checkpoint['state_dict'])
        self.set_settings(checkpoint['settings'])
        print("Model & Settings Successfully Loaded")
        return self
