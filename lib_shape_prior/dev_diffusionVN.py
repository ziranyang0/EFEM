import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm, trange
from copy import deepcopy
import wandb


def imshow(img, file_name=None):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)  # Save the image as a file
    plt.show()  # Display the image

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

@torch.no_grad()
def sample_latent(model, autoencoder, x, steps, eta, classes):
    """Draws samples from a model given starting noise."""
    x = autoencoder.encode(x)
    # global std_infer, mean_infer
    # x = (x-mean_infer)/std_infer
    
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * log_snrs[i], classes).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

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
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # pred = pred*std_infer + mean_infer
    # If we are on the last timestep, output the denoised image
    y = autoencoder.decode(pred)
    return y


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


from vec_layers_out import VecLinearNormalizeActivate as VecLNA
from vec_layers_out import VecLinear
class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims, max_freq, num_bands, std=0.2):
        super(LatentDiffusionModel, self).__init__()
        self.latent_dim = latent_dim
        self.timestep_embed = FourierFeatures(max_freq, num_bands, std=std)
        self.t_emb_dim = 2*num_bands
        
        # input_dim = latent_dim[0] * latent_dim[1] + self.timestep_embed.num_bands * 2  # Additional dims for time embedding
        leak_neg_slope=0.2
        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(VecLNA(self.latent_dim+self.t_emb_dim, hidden_dims[0], mode="so3", act_func=act_func))
            else:
                self.layers.append(VecLNA(hidden_dims[i-1], hidden_dims[i], mode="so3", act_func=act_func))

        # 最后一层输出与原始潜在维度相同
        # self.layers.append(VecLNA(hidden_dims[-1], self.latent_dim, mode="so3", act_func=act_func))
        self.layers.append(VecLinear(v_in=hidden_dims[-1], v_out=self.latent_dim, mode="so3"))

    def forward(self, x, t):
        batch_size = x.size(0)
        # Embed time
        t_emb = self.timestep_embed(t).unsqueeze(2).repeat(1, 1, 3)
        # t_emb = t.unsqueeze(1).unsqueeze(1).repeat(1, self.t_emb_dim, 3)
        # Concatenate time embedding with the latent code
        x = torch.cat((x, t_emb), dim=1)

        # 显式地遍历层，并在某些层上应用残差连接
        for i, layer in enumerate(self.layers):
            # if i > 0 and isinstance(layer, nn.Linear) and x.size(1) == layer.out_features:
            #     # 应用残差连接
            #     x = x + layer(x)
            # else:
            x = layer(x)
        
        # Reshape the output to the original shape
        return x.view(batch_size, self.latent_dim,3)





def eval_loss(model, autoencoder, rng, reals, device):

    # Draw uniformly distributed continuous timesteps
    t = rng.draw(reals.shape[0])[:, 0].to(device)

    # Calculate the noise schedule parameters for those timesteps
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)
    weights = log_snrs.exp() / log_snrs.exp().add(1)

    # Combine the ground truth images and the noise
    alphas = alphas[:, None, None]
    sigmas = sigmas[:, None, None]
    noise = torch.randn_like(reals)
    noised_reals = reals * alphas + noise * sigmas
    targets = noise * alphas - reals * sigmas

    # Compute the model output and the loss.
    with torch.cuda.amp.autocast():
        v = model(noised_reals, log_snrs)
        return (v - targets).pow(2).mean([1, 2]).mul(weights).mean()


def train(model, model_ema, autoencoder, opt, scheduler, rng, train_dl,scaler, epoch, ema_decay, device):
    for i, (reals, pcl) in enumerate(tqdm(train_dl)):
        opt.zero_grad()
        reals = reals.to(device)
        pcl = pcl.to(device)

        # Evaluate the loss
        loss = eval_loss(model, autoencoder, rng, reals, device)

        # Do the optimizer step and EMA update
        scaler.scale(loss).backward()
        scaler.step(opt)
        ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
        scaler.update()

        if i % 50 == 0:
            tqdm.write(f'Epoch: {epoch}, iteration: {i}, loss: {loss.item():g}')
            wandb.log({"epoch": epoch, "iteration": i,
                       "loss": loss.item(),
                       "lr": scheduler.get_last_lr()[0],})
    scheduler.step()

# @eval_mode(model_ema)
@torch.no_grad()
@torch.random.fork_rng()
def val(model_ema, autoencoder, val_dl, device, seed, epoch):
    model_ema.eval()
    autoencoder.eval()
    tqdm.write('\nValidating...')
    torch.manual_seed(seed)
    rng = torch.quasirandom.SobolEngine(1, scramble=True)
    total_loss = 0
    count = 0
    for i, (reals, pcl) in enumerate(tqdm(val_dl)):
        reals = reals.to(device)
        pcl = pcl.to(device)

        loss = eval_loss(model_ema, autoencoder, rng, reals, device)

        total_loss += loss.item() * len(reals)
        count += len(reals)
    loss = total_loss / count
    tqdm.write(f'Validation: Epoch: {epoch}, loss: {loss:g}')

def save(model, model_ema, opt, scaler, epoch):
    # if not os.path.exists('./ckpts'):
    #     os.mkdir('./ckpts')
    filename = '/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt/latent_diffusionVN_test.pth'
    obj = {
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }
    torch.save(obj, filename)



import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, codebook):
        # feat = torch.cat([codebook['z_so3'], #[B, 256, 3]
        #           codebook['z_inv'].unsqueeze(2).repeat(1,1,3), # [B,256] -> [B, 256, 3]
        #           codebook['scale'].unsqueeze(1).unsqueeze(1).repeat(1,1,3), # [B]-> [B,1,3]
        #           codebook['center']], dim=1) # [B, 1, 3]
        feat = torch.cat([codebook['z_so3'], #[B, 256, 3]
                  codebook['z_inv'].unsqueeze(2).repeat(1,1,3)], dim=1) # [B,256] -> [B, 256, 3]
                #   codebook['scale'].unsqueeze(1).unsqueeze(1).repeat(1,1,3), # [B]-> [B,1,3]
                #   codebook['center'].transpose(1,2).repeat(1,1,3)], dim=1) # [B, 3, 3]
                # [B, 1, 3] -> [x,y,z]
        # invariance scalar
        assert feat.shape[1:] == (512,3)
        self.feat = [feat[i] for i in range(feat.shape[0])]
        self.pcl = [codebook['pcl'][i] for i in range(codebook['pcl'].shape[0])]

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        feature = self.feat[idx]
        pcl_item = self.pcl[idx]
        return feature, pcl_item

# N[]
def main():
    # Prepare the dataset
    codebook_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt/codebook.npz"
    with np.load(codebook_path) as data:
        # 将 npz 文件内容转换为字典
        codebook = {key: data[key] for key in data}

    for k, v in codebook.items():
        if isinstance(v, np.ndarray):
            newv = torch.from_numpy(v)
            codebook[k] = newv
        print(k, v.shape)
    # 创建 Dataset
    train_ds = CustomDataset(codebook)

    # 创建 DataLoader
    train_dl = DataLoader(train_ds, batch_size=149, shuffle=True)

    # Create the model and optimizer

    seed = 0

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(0)

    # 更新模型的潜在维度和隐藏层维度
    latent_dim = train_ds[0][0].shape[0]
    # hidden_dims = [2048,4096, 8192, 8192,4096, 2048]  # Example of hidden dimensions
    hidden_dims = [4096, 8192, 8192,4096]  # Example of hidden dimensions
    # hidden_dims = [2048,4096,4096,2048]  # Example of hidden dimensions
    # hidden_dims = [1024,2048,2048,1024]  # Example of hidden dimensions
    # hidden_dims = [2048,2048]  # Example of hidden dimensions
    # hidden_dims = [4096,4096]  # Example of hidden dimensions
    max_freq = 4  # Example max frequency for Fourier features
    num_bands = 4  # Number of frequency bands

    # 初始化模型
    model = LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    model_ema = deepcopy(model)
    # wandb.init(project="vnDiffusion", entity="_zrrr", name="run_vnVanilla_1024_2048_2048_1024")
    wandb.init(project="vnDiffusion", entity="_zrrr", name="run_vnVanilla_4096_8192_lr3e-4_decay0.9_VecLinear")

    print('Diffusion Model parameters:', sum(p.numel() for p in model.parameters()))
    wandb.log({"max_freq": max_freq, "num_bands": num_bands,
            #    "device": device,
               "model_params": sum(p.numel() for p in model.parameters())})
    autoencoder = None
    

    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.8)
    scaler = torch.cuda.amp.GradScaler()
    epoch = 0

    # Use a low discrepancy quasi-random sequence to sample uniformly distributed
    # timesteps. This considerably reduces the between-batch variance of the loss.
    rng = torch.quasirandom.SobolEngine(1, scramble=True)

    # Actually train the model

    ema_decay = 0.998

    # The number of timesteps to use when sampling
    steps = 1000

    # The amount of noise to add each timestep when sampling
    # 0 = no noise (DDIM)
    # 1 = full noise (DDPM)
    eta = 1.
    
    ########### Main Loop ###########
    # val(model_ema, autoencoder, val_dl, device, seed, epoch)
    # demo(model_ema, autoencoder, val_dl, device, seed, epoch, steps, eta)
    while True:
        print('Epoch', epoch)
        train(model, model_ema, autoencoder, opt, scheduler, rng, train_dl, scaler, epoch, ema_decay, device)
        epoch += 1
        # if epoch % 10 == 0:
        #     val(model_ema, autoencoder, val_dl, device, seed, epoch)
        #     demo(model_ema, autoencoder, val_dl, device, seed, epoch, steps, eta)
        #     save(model, model_ema, opt, scaler, epoch)
        if epoch >= 60000:
            break
    save(model, model_ema, opt, scaler, epoch)
    
    return


if __name__ == "__main__":
    main()
