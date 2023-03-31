
import argparse
import torchvision

import torch
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F

# modules
from model import SimpleUnet
from dataloader import load_transformed_dataset
from forward_diffusion import ForwardDiffusion

def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, default='DDPM')
    p.add_argument('--img_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--T', type=int, default=300)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    config = p.parse_args()

    return config



def main(cfg):    

    """
    Data Loader
    """
    data = load_transformed_dataset(cfg)
    dataloader = DataLoader(
            data, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            drop_last=True
        )

    """
    forward diffusion
    """
    def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        betas = torch.linspace(start, end, timesteps)
        return betas
    
    betas = linear_beta_schedule(timesteps = cfg.T, start = 0.0001, end = 0.02)

    forward_diffusion = ForwardDiffusion(cfg.T, betas, cfg.device)


    """
    backward process
    """
    model = SimpleUnet().to(cfg.device)
    
    optimizer = Adam(model.parameters(), lr=0.001)


    """
    Training
    """
    for epoch in tqdm(range(cfg.n_epochs)):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, cfg.T, (cfg.batch_size,), device=cfg.device).long()
            loss = forward_diffusion.get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")


if __name__ == "__main__":

    cfg = define_argparser()

    main(cfg)









