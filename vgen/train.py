"""Training of the UNet video generation GAN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from torch.utils.data import DataLoader
from vgen.videods import MomentsInTimeDataset
from vgen.models import VideoGeneratorBaseline, VideoDiscriminatorBaseline, ImageGeneratorBaseline
from tensorboard_logger import configure, log_value

def setup():

    restore = False
    gen_path = '../data/gen.ckpt'
    disc_path = '../data/disc.ckpt'
    n_frames = 1
    nc_z = 8
    device = torch.device('cuda:2')

    ds = MomentsInTimeDataset('../data/momentsintime/training', max_examples=None, single_example=True)
    init = ImageGeneratorBaseline(nc_z, 8)  # these sizes should be a mismatch?
    gen = VideoGeneratorBaseline(nc_z)
    disc = VideoDiscriminatorBaseline()

    if restore and gen_path is not None:
        print('Restoring generator from save')
        gen.load_state_dict(torch.load(gen_path))

    if restore and disc_path is not None:
        print('Restoring discriminator from save')
        disc.load_state_dict(torch.load(disc_path))

    # set up Tensorboard visualization
    run = '../data/vgen'
    run_path = os.path.join(run, str(datetime.now()))
    if run is not None:
        configure(run_path, flush_secs=5)

    return ds, gen, disc, init, nc_z, n_frames, device, gen_path, disc_path

def train(ds, gen, disc, init, nc_z, device, n_frames, epochs=1, batch_size=4, g_lr=0.0001, d_lr=0.0001, n_disc_trains=3, gen_path=None, disc_path=None, tqdm_disable=False, n_slices=1):
    # CHANGED LEARNING RATE
    gen.to(device)
    disc.to(device)
    init.to(device)
    gen.train()
    disc.train()
    init.train()
    # CHANGING GENERATOR AND DISCRIMINATOR STRUCTURE
    downsample = nn.AvgPool2d(16, stride=16, padding=0)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=g_lr, betas=(0.99, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.99, 0.999))
    gen_losses = []
    disc_losses = []
    for e_idx in range(epochs):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=5)
        bar = tqdm(disable=tqdm_disable, total=len(dl) * n_slices)
        for idx, batch in enumerate(dl):
            batch = [t.to(device) for t in batch]
            label, all_frames = batch  # ignore label for now
            all_frames = all_frames[:, :n_frames].float()
            all_frames = downsample(all_frames.view(-1, 3, 256, 256)).view(batch_size, n_frames, 3, 16, 16)  # downsample to 128x128 per frame
            rand_input = torch.randn(batch_size, nc_z, 16, 16).to(device)

            pyramid = init(rand_input)  # initial the first frame feature pyramid to begin generating

            epsilon = 10e-5

            d_slice = n_frames // n_slices
            for i in range(n_slices):
                bar.update(1)
                real_frames = all_frames[:, i*d_slice: i*d_slice+d_slice]

                log_idx = e_idx * len(bar) + idx
                # load batch of examples
                # we use vanilla GAN objective

                gen_frames_tanh, pyramid = gen(pyramid, n_frames=d_slice)
                fake_scores = disc(gen_frames_tanh)
                log_value('fake_score', torch.mean(fake_scores).item(), log_idx)
                log_value('gen_frame_mean', torch.mean(gen_frames_tanh).item(), log_idx)
                log_value('gen_frame_std', torch.std(gen_frames_tanh).item(), log_idx)
                # train generator less than discriminator
                if idx % (n_disc_trains + 1) == 0:
                    #gen_loss = torch.mean((1-fake_scores)** 2)
                    # real_frames_tanh = (real_frames - 1 / 2) * 2
                    # gen_loss = torch.mean((gen_frames_tanh - real_frames_tanh) ** 2)
                    gen_loss = (1+(-fake_scores).exp() + epsilon).log()
                    item_loss = gen_loss.item()
                    log_value('gen_loss', item_loss, log_idx)
                    gen_losses.append(item_loss)
                    opt_gen.zero_grad()
                    gen_loss.backward()
                    opt_gen.step()
                else:
                    # frames are in range (0, 1), convert to (-1, +1)
                    real_frames_tanh = (real_frames - 1/2) * 2  # convert from sigmoid to tanh
                    log_value('real_frame_mean', torch.mean(real_frames_tanh).item(), log_idx)
                    log_value('real_frame_std', torch.std(real_frames_tanh).item(), log_idx)
                    real_scores = disc(real_frames_tanh)
                    log_value('real_score', torch.mean(real_scores).item(), log_idx)
                    # objective for least-squares GAN
                    #disc_loss = .5 * torch.mean(fake_scores ** 2) + .5 * torch.mean((1-real_scores) ** 2)
                    disc_loss = (1+(-real_scores).exp()+epsilon).log() + (1+fake_scores.exp()+epsilon).log()
                    item_loss = disc_loss.item()
                    log_value('disc_loss', item_loss, log_idx)
                    disc_losses.append(item_loss)
                    opt_disc.zero_grad()
                    disc_loss.backward()
                    opt_disc.step()

                pyramid = [layer.detach() for layer in pyramid]  # we don't want gradients flowing backward

                if log_idx % 10 == 9:
                    if len(gen_losses) > 0 and len(disc_losses) > 0:
                        bar.set_description('gen, disc losses: %s,%s' % (np.mean(gen_losses), np.mean(disc_losses)))
                        gen_losses = []
                        disc_losses = []

                if log_idx % 100 == 99:
                    # periodically save models
                    if gen_path is not None:
                        torch.save(gen.state_dict(), gen_path)
                    if disc_path is not None:
                        torch.save(disc.state_dict(), disc_path)

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    ds, gen, disc, init, nc_z, n_frames, device, gen_path, disc_path = setup()
    train(ds, gen, disc, init, nc_z, device, n_frames, gen_path=gen_path, disc_path=disc_path, batch_size=1, epochs=1,
          tqdm_disable=False)