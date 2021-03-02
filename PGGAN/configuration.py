from argparse import Namespace
import torch
try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

directory = "./Images/"
batch_sizes = [16, 16, 16, 16, 16, 16, 14, 8, 3]  # Number of images per batch at each scale
ngpu = 4
batch_sizes = [batch*ngpu for batch in batch_sizes]
depthScales = [512, 512, 512, 256, 128, 64, 32, 16]
scale_iters = [8e5, 16e5, 16e5, 16e5, 16e5, 16e5, 16e5, 16e5, 2e6]  # Number of images to be shown at each scale
alpha_img_jumps = [0, 8e5, 8e5, 8e5, 8e5, 8e5, 8e5, 8e5, 8e5]  # Allows for alpha mixing for 800k images


args = {
    'name': "ProGAN",
    'batch_sizes': batch_sizes,
    'lr': 0.001,
    'b1': 0,
    'b2': 0.99,
    'eps': 10e-8,
    'lambda_val': 10,
    'drift': 0.001,
    'z_dim': 512,
    'directory': directory,
    'worker_num': 0,
    'g_norm_layers': True,
    'd_norm_layers': False,
    'init_bias': True,
    'equalize': True,
    'leakiness': 0.2,
    'mode': 'nearest',
    'img_channels': 3,
    'initial_channel_depth': 512,
    'minibatch_std': True,
    'decision_dim': 1,
    'iterations': scale_iters,
    'depths': depthScales,
    'alpha_jumps': alpha_img_jumps,
    'sample': 300,  # Draw a sample once every 100 iterations
    'sample_num': 16,
    'ema': True,
    'ema_decay': 0.999,
    'set_fp16': False
}

if args['set_fp16'] and not APEX_AVAILABLE:
    raise Exception("Mixed Precision Training Selected but Apex Not Available")

# Storing the hyperparameters doesn't support lists -> Need to convert to tensors
for key, value in args.items():
    if type(value).__name__ == 'list':
        args[key] = torch.Tensor(value)

hparams = Namespace(**args)
