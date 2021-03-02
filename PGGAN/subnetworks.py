import copy
import torch
from GAN.network import ProDiscriminator


class FeatureExtractor(ProDiscriminator):
    def __init__(self, checkpoint):
        state_dict, settings = checkpoint['state_dict'], checkpoint['settings']
        super(FeatureExtractor, self).__init__(channel_depth=settings['channel_depth'][0],
                                               init_bias=settings['init_bias'],
                                               norm_layers=settings['norm_layers'],
                                               input_channels=settings['input_channels'],
                                               decision_layer_dim=settings['decision_layer_dim'],
                                               equalize_layers=settings['equalize'],
                                               leakiness=settings['leakiness'],
                                               minibatch_std=settings['minibatch'])
        self.load(checkpoint)
        del self.DecisionLinear  # Remove duplicate final scoring layer

    # def forward(self, x, pre_classify=False):
    #     batch, channels, height, width = x.shape
    #     extract = []  # Multi-Scale Feature Extraction
    #     # Convert from RGB to Feature Maps:
    #     x = self.fromRGB[-1](x)
    #
    #     for scale_num, scale_block in enumerate(self.ScaleBlocks):
    #         x = scale_block(x)  # Process the input through the contraction block of conv, conv, downsample
    #
    #     if self.minibatch:
    #         x = self.miniSTD(x, dim=1)  # Calculate the std across the feature maps and not minibatchs
    #
    #     x = self.DecisionBlock(x)  # Final 3x3, 4x4 conv
    #     x = x.view(batch, -1)
    #     y = self.classifier(x)
    #     if pre_classify:
    #         # Return both the final classification and pre-classification representation
    #         return y, x
    #     else:
    #         return y

    def forward(self, x):#, map_extracts=1, cat=True):
        batch, channels, height, width = x.shape
        extract = []  # Multi-Scale Feature Extraction
        # Convert from RGB to Feature Maps:
        x = self.fromRGB[-1](x)

        for scale_num, scale_block in enumerate(self.ScaleBlocks):
            # if len(self.ScaleBlocks) - scale_num <= (map_extracts-2):  # Remove 2 due to the final 2 feature maps
            #     extract.append(x)
            x = scale_block(x)  # Process the input through the contraction block of conv, conv, downsample

        # if map_extracts >= 2:
        #     extract.append(x)

        if self.minibatch:
            x = self.miniSTD(x, dim=1)  # Calculate the std across the feature maps and not minibatchs

        x = self.DecisionBlock(x)  # Final 3x3, 4x4 conv

        x = x.view(batch, -1)

        # if map_extracts >= 1:
        #     extract.append(x)

        # if cat:
        #     extract = torch.cat([feat.view(batch, -1) for feat in extract], dim=1)
        return x#extract  # Final set of extracted feature vectors for an image
