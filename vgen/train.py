"""Training of the UNet video generation GAN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from vgen.videods import MomentsInTimeDataset
from vgen.models import VideoGeneratorBaseline, VideoDiscriminatorBaseline

def setup():
    n_frames = 30
    nc_z = 8
    device =torch.device('cuda:0')
    ds = MomentsInTimeDataset('../data/momentsintime/training')
    gen = VideoGeneratorBaseline(n_frames, nc_z)
    disc = VideoDiscriminatorBaseline()

    return ds, gen, disc, nc_z, n_frames, device

def train(ds, gen, disc, nc_z, device, n_frames, batch_size=8, lr=0.0001, num_disc_trains=5):
    gen.to(device)
    disc.to(device)
    gen.train()
    disc.train()
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.99, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.99, 0.999))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=5)
    bar = tqdm(dl)
    gen_losses = []
    disc_losses = []
    for idx, batch in enumerate(bar):
        # load batch of examples
        batch = [t.to(device) for t in batch]
        label, real_frames = batch # ignore label for now
        real_frames = real_frames[:, :n_frames].float()
        rand_input = torch.rand(batch_size, nc_z, 64, 64).to(device)
        gen_frames_tanh = gen(rand_input)
        fake_scores = disc(gen_frames_tanh)

        # train generator less than discriminator
        if idx % num_disc_trains == 0:
            gen_loss = torch.mean((1-fake_scores) ** 2)
            gen_losses.append(gen_loss.item())
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()
        else:
            real_frames_tanh = (real_frames - 1/2) * 2  # convert from sigmoid to tanh
            real_scores = disc(real_frames_tanh)
            disc_loss = .5 * torch.mean(fake_scores ** 2) + .5 * torch.mean((1-real_scores) ** 2)
            disc_losses.append(disc_loss.item())
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

        if idx % 10 == 9:
            if len(gen_losses) > 0 and len(disc_losses) > 0:
                bar.set_description('gen, disc losses: %s,%s' % (np.mean(gen_losses), np.mean(disc_losses)))
                gen_losses = []
                disc_losses = []

if __name__ == '__main__':
    ds, gen, disc, nc_z, n_frames, device = setup()
    train(ds, gen, disc, nc_z, device, n_frames)