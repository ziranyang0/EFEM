import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import wandb
import argparse
from torch.utils.data import Dataset, DataLoader

from dev_utils import ema_update, get_alphas_sigmas, get_ddpm_schedule, FourierFeatures, save_model

from vec_layers_out import VecLinearNormalizeActivate as VecLNA
from vec_layers_out import VecLinear
class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims, scalar_hidden_dims, max_freq, num_bands, std=0.2):
        super(LatentDiffusionModel, self).__init__()
        self.latent_dim = latent_dim
        self.scalar_hidden_dims = scalar_hidden_dims
        self.timestep_embed = FourierFeatures(max_freq, num_bands, std=std)
        self.t_emb_dim = 2*num_bands
        
        leak_neg_slope=0.2
        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)
        self.layers = nn.ModuleList()
        self.skip_connections = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(VecLNA(in_features=self.latent_dim+self.t_emb_dim,
                                          out_features=hidden_dims[0],
                                          s_in_features=self.latent_dim+self.t_emb_dim,
                                          s_out_features=scalar_hidden_dims[0],
                                          mode="so3", 
                                        #   s_normalization=torch.nn.BatchNorm1d(scalar_hidden_dims[0]),
                                          act_func=act_func))
                if self.latent_dim + self.t_emb_dim == hidden_dims[0] and self.latent_dim + self.t_emb_dim == scalar_hidden_dims[0]:
                    self.skip_connections.append(True)
                else:
                    self.skip_connections.append(False)
            else:
                self.layers.append(VecLNA(in_features=hidden_dims[i-1],
                                          out_features=hidden_dims[i],
                                          s_in_features=scalar_hidden_dims[i-1],
                                          s_out_features=scalar_hidden_dims[i],
                                        #   s_normalization=torch.nn.BatchNorm1d(scalar_hidden_dims[i]),
                                          mode="so3", act_func=act_func))
                if hidden_dims[i-1] == hidden_dims[i] and scalar_hidden_dims[i-1] == scalar_hidden_dims[i]:
                    self.skip_connections.append(True)
                else:
                    self.skip_connections.append(False)
        
        self.vn_head = VecLinear(v_in=hidden_dims[-1],
                                 v_out=self.latent_dim,
                                 s_in=scalar_hidden_dims[-1],
                                 s_out=self.latent_dim,
                                 mode="so3",)
        # self.scalar_head = nn.Linear(scalar_hidden_dims[-1], self.latent_dim)
    
    def forward(self, z_so3, z_inv, t):
        batch_size = z_so3.size(0)
        # Embed time
        t_emb_s = self.timestep_embed(t)
        t_emb_x = self.timestep_embed(t).unsqueeze(2).repeat(1, 1, 3)
        z_inv = torch.cat([z_inv, t_emb_s], dim=1)
        z_so3 = torch.cat([z_so3, t_emb_x], dim=1)

        for i, layer in enumerate(self.layers):
            z_so3_new, z_inv_new = layer(z_so3, z_inv)
            if self.skip_connections[i]:
                z_so3 = z_so3_new + z_so3
                z_inv = z_inv_new + z_inv
            else:
                z_so3 = z_so3_new
                z_inv = z_inv_new
        
        # pred_z_so3 = self.vn_head(z_so3)
        # pred_z_inv = self.scalar_head(z_inv)
        pred_z_so3, pred_z_inv = self.vn_head(z_so3, z_inv)
        
        return pred_z_so3, pred_z_inv

class Residual_LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims, max_freq, num_bands, std=0.2):
        super(Residual_LatentDiffusionModel, self).__init__()
        self.latent_dim = latent_dim * 4
        self.timestep_embed = FourierFeatures(max_freq, num_bands, std=std)
        self.t_emb_dim = 2 * num_bands

        self.layers = nn.ModuleList()
        leak_neg_slope = 0.2
        self.act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)
        self.skip_connections = []

        for i, hidden_dim in enumerate(hidden_dims):
            in_dim = self.latent_dim + self.t_emb_dim if i == 0 else hidden_dims[i-1]
            out_dim = hidden_dims[i]
            self.layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))

            # Check if dimensions match for a skip-connection
            if in_dim == out_dim:
                self.skip_connections.append(True)
            else:
                self.skip_connections.append(False)

        # 最后一层输出与原始维度相同
        self.layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=self.latent_dim))
        self.skip_connections.append(False)

    def forward(self, z_so3, z_inv, t):
        batch_size = z_so3.size(0)
        # Embed time
        t_emb = self.timestep_embed(t)
        z_so3 = z_so3.reshape(batch_size, -1)  # [b, 256, 3] -> [b, 768]
        feat = torch.cat([z_so3, z_inv, t_emb], dim=1)  # [b, 768+256+xxx]

        for i, layer in enumerate(self.layers):
            if self.skip_connections[i]:
                feat = self.act_func(layer(feat)) + feat  # Residual connection
            else:
                feat = self.act_func(layer(feat))

        pred_z_so3 = feat[:, :3*256].reshape(batch_size, -1, 3)
        pred_z_inv = feat[:, 3*256:]
        
        return pred_z_so3, pred_z_inv


def train(model, model_ema, opt, scheduler, rng, train_dl, 
          scaler, epoch, ema_decay, device, args):
    # for i, ((x, s), pcl) in enumerate(tqdm(train_dl)):
    for i, ((x, s), pcl) in enumerate(train_dl):
        
        opt.zero_grad()
        pcl = pcl.to(device)
        x = x.to(device)
        s = s.to(device)

        # Evaluate the loss

        # Draw uniformly distributed continuous timesteps
        t = rng.draw(x.shape[0])[:, 0].to(device)

        # Calculate the noise schedule parameters for those timesteps
        log_snrs = get_ddpm_schedule(t)
        alphas, sigmas = get_alphas_sigmas(log_snrs)
        weights = log_snrs.exp() / log_snrs.exp().add(1)

        # Combine the ground truth images and the noise
        alphas_x = alphas[:, None, None]
        sigmas_x = sigmas[:, None, None]
        noise_x = torch.randn_like(x)
        noised_x = x * alphas_x + noise_x * sigmas_x
        target_x = noise_x * alphas_x - x * sigmas_x

        alphas_s = alphas[:, None]
        sigmas_s = sigmas[:, None]
        noise_s = torch.randn_like(s)
        noised_s = s * alphas_s + noise_s * sigmas_s
        target_s = noise_s * alphas_s - s * sigmas_s

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            pred_x, pred_s= model(noised_x, noised_s, log_snrs)
            # (pred_x - target_x).pow(2).shape: [149, 256, 3]
            # (pred_s - target_s).pow(2).shape: [149, 256]
            loss =  ((pred_x - target_x).pow(2).mean([1, 2]).mul(weights).mean()*1/2 + \
                    (pred_s - target_s).pow(2).mean(1).mul(weights).mean()*1/2) 
    

        # Do the optimizer step and EMA update
        scaler.scale(loss).backward()
        scaler.step(opt)
        ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
        scaler.update()

        if i % args.log_interval == 0 and epoch %1 == 0:
            print(f'Epoch: {epoch}, iteration: {i}, loss: {loss.item():g}')
            if not args.debug:
                wandb.log({"epoch": epoch, "iteration": i,
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],})
        
    scheduler.step()

class CustomDataset(Dataset):
    def __init__(self, codebook_path):
        with np.load(codebook_path) as data:
            # 将 npz 文件内容转换为字典
            codebook = {key: data[key] for key in data}
        del codebook['id']
        for k, v in codebook.items():
            if isinstance(v, np.ndarray):
                newv = torch.from_numpy(v)
                codebook[k] = newv
            print(k, v.shape)
        
        self.x = [codebook['z_so3'][i] for i in range(codebook['z_so3'].shape[0])] # [256, 3]
        self.s = [codebook['z_inv'][i] for i in range(codebook['z_inv'].shape[0])] # [256]
        self.pcl = [codebook['pcl'][i] for i in range(codebook['pcl'].shape[0])]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        s = self.s[idx]
        pcl_item = self.pcl[idx]
        return (x, s), pcl_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_interval", default=10000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--category", default="mugs", type=str)
    parser.add_argument("--bs", default=512, type=int)
    parser.add_argument("--exp_name", default="run", type=str)
    parser.add_argument("--log_interval", default=1, type=int)
    args = parser.parse_args()

    if args.debug:
        print("Debug mode is enabled")

    args.exp_name = "chairs_vnhead_residual_2048*4_cos_10k"
    args.category = "chairs"
    args.num_epochs = 10000
    assert args.category in ["mugs", "chairs", "kit4cates"]
    # Prepare the dataset
    # codebook_path = "/home/ziran/se3/EFEM/lib_shape_prior/dev_ckpt/codebook.npz"
    # codebook_path = "/home/ziran/se3/EFEM/cache/mugs.npz"
    # codebook_path = "/home/ziran/se3/EFEM/cache/kit4cates.npz"
    codebook_path = f"/home/ziran/se3/EFEM/cache/{args.category}.npz"
    train_ds = CustomDataset(codebook_path)

    # 创建 DataLoader
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)


    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)


    latent_dim = 256
    hidden_dims = [2048, 2048, 2048, 2048]
    # hidden_dims = [4096, 8192, 8192, 4096]  # Example of hidden dimensions
    # hidden_dims = [4096, 4096, 4096, 4096]  # Example of hidden dimensions
    max_freq = 4  # Example max frequency for Fourier features
    num_bands = 4  # Number of frequency bands

    # hidden_dims = [256 * 16, 4096, 4096, 2048]
    # model = CNN_MLP_diffusion(latent_dim, hidden_dims, max_freq, num_bands).to(device)
        # wandb.init(project="vnDiffusion", entity="_zrrr", name="test_CNNMLP")

    # 初始化模型
    # model = origin_LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    # model = Residual_LatentDiffusionModel(latent_dim, hidden_dims, max_freq, num_bands).to(device)
    # if not args.debug:
    #     wandb.init(project="vnDiffusion", entity="_zrrr", name="test_residualMLP_4096*4")
    #     # wandb.init(project="vnDiffusion", entity="_zrrr", name="origin_4096_8192*2_weighted1/2_leakyrelu_cos_200k")

    scalar_hidden_dims = [256,256,256,256]
    model = LatentDiffusionModel(latent_dim, hidden_dims, scalar_hidden_dims, max_freq, num_bands).to(device)
    if not args.debug:
        wandb.init(project="vnDiffusion", entity="_zrrr", name=args.exp_name)
    
    print('Diffusion Model parameters:', sum(p.numel() for p in model.parameters()))
    if not args.debug:
        wandb.log({"max_freq": max_freq, "num_bands": num_bands,
                #    "device": device,
                "model_params": sum(p.numel() for p in model.parameters())})
    
               
    model_ema = deepcopy(model)
    

    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_dl)*args.num_epochs, eta_min=1e-7)
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
    while True:
        # print('Epoch', epoch)
        train(model, model_ema, opt, scheduler, rng, train_dl, scaler, epoch, ema_decay, device, args)
        epoch += 1
        if (epoch + 1) % args.save_interval == 0:
            save_model(model, model_ema, opt, scaler, epoch, args)
        if epoch >= args.num_epochs:
            break
    
    return


if __name__ == "__main__":
    main()
