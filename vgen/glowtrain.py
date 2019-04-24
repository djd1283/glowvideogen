"""Used from https://github.com/rosinality/glow-pytorch for class project. Thank you!"""

from shutil import rmtree
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from vgen.glowmodel import Glow
from vgen.videods import MomentsInTimeDataset, write_video
from tensorboard_logger import log_value, configure
from tgalert import TelegramAlert
import datetime

alert = TelegramAlert()

configure('logdir/' + str(datetime.datetime.now()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=16, type=int, help='batch size')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to train model')
parser.add_argument(
    '--n_flow', default=32, type=int, help='number of flows in each block'
)
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument(
    '--no_lu',
    action='store_true',
    help='use plain convolution instead of LU decomposed version',
)
parser.add_argument(
    '--affine', action='store_true', help='use affine coupling instead of additive'
)
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
parser.add_argument('--path', metavar='PATH', type=str, help='Path to image directory')
parser.add_argument('--video_path', metavar='PATH', type=str, help='Where to save generated video samples')
parser.add_argument('--n_frames', default=90, type=int, help='max number of video frames')
parser.add_argument('--max_examples', default=None, type=int, help='max number of examples to train on')

# SET TEMPERATURE TO 1 WHEN MODELLING ONLY ONE IMAGE - IT SHOULD BE AT THE CENTER

def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def generate_image(model, img_size, n_flow, n_block, n_sample, temp=0.7):
    # Determine sizes of each layer
    z_sample = []
    z_shapes = calc_z_shapes(3, img_size, n_flow, n_block)
    for z in z_shapes:
        z_new = torch.randn(n_sample, *z) * temp
        z_sample.append(z_new.to(device))
    return model.reverse(z_sample).cpu().data


def generate_video(head_model, tail_model, n_frames, img_size, n_flow, n_block, n_sample, temp=0.7):
    image = generate_image(head_model, img_size, n_flow, n_block, n_sample, temp)
    frames = [image]
    for i in range(n_frames-1):
        frame = generate_image(tail_model, img_size, n_flow, n_block, n_sample, temp)
        frames.append(frame)
    return torch.stack(frames, dim=1)


def train(args, head_model, tail_model, n_epochs=1, n_frames=90, max_examples=None):
    """Head model produces the first frame of the video. Tail model produces the rest of the frames."""
    # Load dataset of images
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    ds = MomentsInTimeDataset('../data/momentsintime/training', max_examples=max_examples)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=3)
    head_optimizer = optim.Adam(head_model.parameters(), lr=args.lr)
    tail_optimizer = optim.Adam(tail_model.parameters(), lr=args.lr)
    down_x = 256 // args.img_size
    downsample = nn.AvgPool2d(down_x, stride=down_x, padding=0).to(device)
    n_bins = 2. ** args.n_bits

    if args.video_path is not None and not os.path.exists(args.video_path):
        os.makedirs(args.video_path)

    pbar = tqdm(total=n_epochs * len(dl))
    for epoch_idx in range(n_epochs):
        for batch_idx, batch in enumerate(range(1000000)):
            pbar.update(1)
            i = epoch_idx * len(dl) + batch_idx
            image_old, _ = next(dataset)
            image_old = image_old.to(device)
            #batch = [t.to(device) for t in batch]
            #label, frames = batch
            #frames = frames - 0.5 # normalize to range (-0.5, +0.5)
            #log_value('video_std', frames.std().item(), i)
            #log_value('video_mean', frames.mean().item(), i)
            #image = frames[:, 0]
            #image = downsample(image)
            image = image_old
            #tail_frames = frames[:, 1:]

            def train_on_frame(model, im, optimizer):
                log_p, logdet = model(im + torch.rand_like(im) / n_bins)
                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                return loss, log_p, log_det

            loss, log_p, log_det = train_on_frame(head_model, image.clone(), head_optimizer)

            log_value('loss', loss.item(), i)
            log_value('log_p', log_p.item(), i)
            log_value('log_det', log_det.item(), i)

            # for i in range(tail_frames.shape[1]):
            #     # Now we generate all the rest of the frames
            #     frame = downsample(tail_frames[:, i])
            #     train_on_frame(tail_model, frame, tail_optimizer)

            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))

            pbar.set_description(
                f'Epoch: {epoch_idx}; loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}'
            )

            if i % 100 == 0 or i == 0:
                with torch.no_grad():
                    gen_image = generate_image(head_model, args.img_size, args.n_flow, args.n_block, args.n_sample,
                                               args.temp)

                    if args.video_path is not None:
                        gen_video = generate_video(head_model, tail_model, n_frames, args.img_size, args.n_flow, args.n_block, args.n_sample,
                                                   args.temp)
                        # Save all generated video samples to disk
                        for j in range(gen_video.shape[0]):
                            write_video(os.path.join(args.video_path, 'video_' + str(j) + '.mp4'), gen_video)

                    utils.save_image(
                        gen_image,
                        f'sample/{str(i + 1).zfill(6)}.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )
                    utils.save_image(
                        image.cpu().data,
                        f'sample/{str(i + 1).zfill(6)}_real.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            # CURRENTLY NOT SAVING TAIL MODEL
            if i % 10000 == 0:
                torch.save(
                    head_model.state_dict(), f'checkpoint/model_{str(i + 1).zfill(6)}.pt'
                )
                torch.save(
                    head_optimizer.state_dict(), f'checkpoint/optim_{str(i + 1).zfill(6)}.pt'
                )

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    rmtree('sample/')
    os.makedirs('sample/')

    head_frame_model = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    head_frame_model = head_frame_model.to(device)

    tail_frame_model = Glow(3, args.n_flow // 4, args.n_block // 2, affine=args.affine, conv_lu=not args.no_lu)
    tail_frame_model = tail_frame_model.to(device)
    #model = nn.DataParallel(model_single)
    # model = model_single

    train(args, head_frame_model, tail_frame_model, n_epochs=args.epochs, max_examples=args.max_examples)

    alert.write('glowtrain.py: Finished training')
