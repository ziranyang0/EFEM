
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm, trange

@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

# Define the noise schedule and sampling loop

def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given the log SNR for a timestep."""
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()

def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.special.expm1(1e-4 + 10 * t**2).log()

class FourierFeatures(nn.Module):
    def __init__(self, max_freq, num_bands, std=1.0):
        super().__init__()
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.std = std

    def forward(self, t):
        freqs = torch.exp2(torch.linspace(0, self.max_freq, self.num_bands, device=t.device))
        steps = t[:, None] * freqs[None]
        embedding = torch.cat((steps.sin(), steps.cos()), dim=-1) * self.std
        return embedding


@torch.no_grad()
def sample_xs(model, x, s, steps, eta):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v_x, v_s = model(x, s, ts * log_snrs[i])

        # Predict the noise and the denoised image
        pred_x = x * alphas[i] - v_x * sigmas[i]
        eps_x = x * sigmas[i] + v_x * alphas[i]
        pred_s = s * alphas[i] - v_s * sigmas[i]
        eps_s = s * sigmas[i] + v_s * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred_x * alphas[i + 1] + eps_x * adjusted_sigma
            s = pred_s * alphas[i + 1] + eps_s * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma
                s += torch.randn_like(s) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred_x, pred_s

import os
def save_model(model, model_ema, opt, scaler, epoch,
         args, exp_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt"):
    if not os.path.exists(f"{exp_path}/{args.exp_name}"):
        os.makedirs(f"{exp_path}/{args.exp_name}")
    filename = f'{exp_path}/{args.exp_name}/model.pth'
    obj = {
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        "category": args.category
    }
    torch.save(obj, filename)



# --------------------------------------------------------------

import matplotlib.pyplot as plt

def plot_schedulers():
    # Commented out IPython magic to ensure Python compatibility.
    # Visualize the noise schedule

    # %config InlineBackend.figure_format = 'retina'
    plt.rcParams['figure.dpi'] = 100

    t_vis = torch.linspace(0, 1, 1000)
    log_snrs_vis = get_ddpm_schedule(t_vis)
    alphas_vis, sigmas_vis = get_alphas_sigmas(log_snrs_vis)

    print('The noise schedule:')

    plt.plot(t_vis, alphas_vis, label='alpha (signal level)')
    plt.plot(t_vis, sigmas_vis, label='sigma (noise level)')
    plt.legend()
    plt.xlabel('timestep')
    plt.grid()
    plt.show()

    plt.plot(t_vis, log_snrs_vis, label='log SNR')
    plt.legend()
    plt.xlabel('timestep')
    plt.grid()
    plt.show()