"""Used from https://github.com/rosinality/glow-pytorch for class project. Thank you!"""

from shutil import rmtree
import os
from tqdm import tqdm
import numpy as np
import glob
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from vgen.glowmodel import Glow, LabelGlow, ContextProcessor
from vgen.videods import MomentsInTimeDataset, write_video
from tensorboard_logger import log_value, configure
from tgalert import TelegramAlert
import datetime

alert = TelegramAlert()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=16, type=int, help='batch size')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to train model')
parser.add_argument(
    '--n_flow', default=32, type=int, help='number of flows in each block'
)
parser.add_argument('--restore', default=False, action='store_true', help='if true, restore from saved checkpoint')
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
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
parser.add_argument('--path', metavar='PATH', type=str, help='Path to image directory')
parser.add_argument('--video_path', metavar='PATH', type=str, help='Where to save generated video samples')
parser.add_argument('--n_frames', default=30, type=int, help='max number of video frames')
parser.add_argument('--max_examples', default=None, type=int, help='max number of examples to train on')
parser.add_argument('--use_state', default=False, action='store_true', help='if true, allow state info to pass between frames')
parser.add_argument('--use_label', default=False, action='store_true', help='if true, allow state info to pass between frames')


args = parser.parse_args()

i = None

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


def calc_num_trainable_params(params):
    model_parameters = filter(lambda p: p.requires_grad, params)
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params

def calc_grad_std_params(params):
    model_parameters = filter(lambda p: p.requires_grad, params)
    squared_sum = sum((p.data ** 2).sum().item() for p in model_parameters)
    return np.sqrt(squared_sum)

def generate_image(model, img_size, n_flow, n_block, n_sample, temp=0.7, ctx=None, label=None):
    """Generate a single image from a Glow model."""
    # Determine sizes of each layer

    z_sample = []
    z_shapes = calc_z_shapes(3, img_size, n_flow, n_block)
    for z in z_shapes:
        z_new = torch.randn(n_sample, *z) * temp
        z_sample.append(z_new.to(device))

    assert ctx is None or label is None  # can either insert label or context
    if label is not None:
        return model.reverse(z_sample, label=label)
    else:
        # handles both cases where only context is provided or no label or context is provided
        return model.reverse(z_sample, ctx=ctx)


def generate_video(head_model, tail_model, ctx_proc, n_frames, img_size, n_flow, n_block, temp=0.7, label=None):
    """Generate video frame by frame, each frame conditioned on the previous frame only"""
    # lookup label embeddings as input to model

    image = generate_image(head_model, img_size, n_flow, n_block, 1, temp, label=label)
    frames = [image]
    state = torch.zeros(1, ctx_proc.nc_state, img_size, img_size).to(device)
    for j in range(n_frames-1):
        prev_frame = frames[-1]
        ctx=prev_frame
        if args.use_state:
            # produce new state given the previous frame
            state = ctx_proc(prev_frame, state)
            state = ctx_proc.norms[j](state)  # normalize each state across all frames
            # add state to context for glow to generate from
            ctx = torch.cat([ctx, state], dim=1)

        frame = generate_image(tail_model, img_size, n_flow // 32, n_block, 1, temp, ctx=ctx)
        #frame = frames[-1] + delta
        frames.append(frame)
    result = torch.stack(frames, dim=1).squeeze(0)
    return result


def train_on_first_frame(model, im, label, optimizer, n_bins):
    """
    Train glow to maximize probability for first frame of videos.
    :param model:
    :param l_embs: (n_labels, nc, img_size, img_size)
    :param im: (batch_size, 3, img_size, img_size
    :param label: (batch_size,)
    :param optimizer: optimizer to step parameters
    :param n_bins: resolution of pixel storage
    :return:
    """
    log_p, logdet = model(im + torch.rand_like(im) / n_bins, label=label)
    loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
    model.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)  # 100 times the parameters, 10 times the norm
    log_value('norm', norm, i)
    if model.l_embs.grad is not None:
        log_value('label_emb_std', model.l_embs.data.std().item(), i)
        log_value('label_emb_grad_std', model.l_embs.grad.std().item(), i)
        log_value('label_emb_grad_norm', model.l_embs.grad.norm(). item(), i)
    optimizer.step()
    return loss, log_p, log_det


def calc_video_loss(tail_model, ctx_proc, frames, n_bins, log_values=False):
    batch_size, n_frames, nc, h, w = frames.shape
    state = torch.zeros(batch_size, ctx_proc.nc_state, h, w).to(frames.device)
    losses = []
    for j in range(frames.shape[1] - 1):
        # now we insert context here
        prev_frame = frames[:, j]
        frame = frames[:, j + 1]
        #delta = frame - prev_frame  # we will produce the difference
        ctx = prev_frame  # we give the previous frame as context

        if args.use_state:
            # produce new state given the previous frame
            state = ctx_proc(prev_frame, state)
            state = ctx_proc.norms[j](state)  # normalize each state across all frames
            if j == 1 and log_values:
                log_value('ctx_std_start', state.std().item(), i)
            if j == frames.shape[1] - 2 and log_values:
                log_value('ctx_std_end', state.std().item(), i)
            # add state to context for glow to generate from
            ctx = torch.cat([ctx, state], dim=1)

        log_p, logdet = tail_model(frame + torch.rand_like(frame) / n_bins, ctx=ctx)
        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        losses.append(loss)
    total_loss = torch.stack(losses, dim=0).mean()
    return total_loss


def train_on_video(head_model, tail_model, ctx_proc, frames, label, head_optimizer, tail_optimizer, n_bins):
    """
    Train glow to generate all frames of a video
    :param head_model: this Glow model generates the first frame of a video
    :param head_frame: this Glow model generates all other frames of the video
    :param frames: frames of the video to train on, tensor (B, N, 3, H, W)
    :param state: initial state to store context from (B, SC, H, W)
    :param optimizer: optimizer to step parameters
    :return: calculated loss for video, followed by gaussian log probability and log determinant
    """
    # train on first frame first
    first_loss, first_log_p, first_log_det = train_on_first_frame(head_model, frames[:, 0],
                                                                  label, head_optimizer, n_bins)
    # now train the rest of the frames as one step
    tail_optimizer.zero_grad()
    total_loss = calc_video_loss(tail_model, ctx_proc, frames, n_bins, log_values=True)
    total_loss.backward()
    tail_norm = torch.nn.utils.clip_grad_norm_(tail_model.parameters(), 5)
    ctx_norm = torch.nn.utils.clip_grad_norm_(ctx_proc.parameters(), 5)
    log_value('video_norm', tail_norm, i)
    log_value('ctx_norm', ctx_norm, i)
    tail_optimizer.step()

    return total_loss, first_loss, first_log_p, first_log_det


def restore_models_optimizers_from_save(head_model, tail_model, ctx_proc, head_optim, tail_optim):
    """Restore all models and optimizers from save."""
    head_model.load_state_dict(torch.load('checkpoint/headmodel.pt'))
    tail_model.load_state_dict(torch.load('checkpoint/tailmodel.pt'))
    ctx_proc.load_state_dict(torch.load('checkpoint/ctxmodel.pt'))
    head_optim.load_state_dict(torch.load('checkpoint/head_optim.pt'))
    tail_optim.load_state_dict(torch.load('checkpoint/tail_optim.pt'))


def train(args, ds, head_model, tail_model, ctx_proc, head_optimizer, tail_optimizer, n_epochs=1, n_frames=30):
    """Head model produces the first frame of the video. Tail model produces the rest of the frames."""
    # Load dataset of images
    #dataset = iter(sample_data(args.path, args.batch, args.img_size))
    dl = DataLoader(ds, batch_size=args.batch, num_workers=3)
    down_x = 256 // args.img_size
    downsample = nn.AvgPool2d(down_x, stride=down_x, padding=0).to(device)
    n_bins = 2. ** args.n_bits
    inf = float('inf')
    head_model.train()
    tail_model.train()
    ctx_proc.train()
    global i  # we keep the iteration number as a global variable, to allow logging values anywhere
    i = 0
    img_count=0
    pbar = tqdm(total=n_epochs * len(dl))
    for epoch_idx in range(n_epochs):
        for batch_idx, batch in enumerate(dl):
            pbar.update(1)
            i = epoch_idx * len(dl) + batch_idx
            #image_old, _ = next(dataset)
            #image_old = image_old.to(device)
            batch = [t.to(device) for t in batch]
            label, frames = batch
            label = label if args.use_label else None
            frames = frames[:, :n_frames]
            frames = frames - 0.5 # normalize to range (-0.5, +0.5)
            # downsample all frames to make them easier to generate
            frames = torch.stack([downsample(frames[:, i]) for i in range(frames.shape[1])], 1)

            log_value('video_std', frames.std().item(), i)
            log_value('video_mean', frames.mean().item(), i)

            image = frames[:, 0]

            #TODO condition on label features
            #TODO add DNI to train on longer videos

            video_loss, loss, log_p, log_det\
                = train_on_video(head_model, tail_model, ctx_proc, frames, label, head_optimizer, tail_optimizer, n_bins)
            #loss, log_p, log_det = train_on_first_frame(head_model, image, head_optimizer, n_bins)

            log_value('loss', loss.item(), i)
            log_value('log_p', log_p.item(), i)
            log_value('log_det', log_det.item(), i)
            log_value('video_loss', video_loss.item(), i)

            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))

            pbar.set_description(
                f'Epoch: {epoch_idx}; loss: {loss.item():.5f}; v_loss: {video_loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}'
            )

            if i % 100 == 0:
                with torch.no_grad():

                    head_model.eval()
                    tail_model.eval()
                    ctx_proc.eval()
                    sample_label = torch.randint(0, n_labels, (args.n_sample,)) if args.use_label else None
                    # TODO fix ctx versus label problem
                    gen_image = generate_image(head_model, args.img_size, args.n_flow, args.n_block, args.n_sample,
                                               args.temp, label=sample_label)
                    gen_image = gen_image.cpu()

                    if args.video_path is not None:
                        sample_label = torch.randint(0, n_labels, (1,))  # generate images from label
                        gen_video = generate_video(head_model, tail_model, ctx_proc, n_frames, args.img_size,
                                                   args.n_flow, args.n_block, args.temp, label=sample_label)
                        gen_video = np.transpose(gen_video.cpu().numpy(), [0, 2, 3, 1])
                        gen_video = ((gen_video + 0.5) * 256.0)
                        gen_video = gen_video.clip(0, 255).astype(np.uint8)
                        # Save all generated video samples to disk
                        write_video(os.path.join(args.video_path, 'video_' + str(img_count) + '.mp4'), gen_video)
                        with open(os.path.join(args.video_path, 'video_' + str(img_count) + '.txt'), 'w') as f:
                            # write the video label for reference
                            f.write('label: %s\n' % str(ds.index_to_label[sample_label.item()]))

                    utils.save_image(
                        gen_image,
                        f'sample/{str(img_count).zfill(4)}.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )
                    utils.save_image(
                        image.cpu().data,
                        f'sample/{str(img_count).zfill(4)}_real.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )
                    img_count += 1

            if i % 100 == 99:
                # NaN fail-safe: if we detect Nan or inf loss, restore previous checkpoint
                # NaN occurs every so often (once after two hours, or not at all) due to numerical instability (division, log, etc.)
                if loss.item() != loss.item() or video_loss.item() != video_loss.item() or loss.abs().item() == inf:
                    alert.write('glowtrain.py: NaN detected! Restoring last save')
                    restore_models_optimizers_from_save(head_model, tail_model, ctx_proc, head_optimizer,
                                                        tail_optimizer)
                else:
                    pbar.write("Saving model. Loss: %s" % loss.item())
                    torch.save(head_model.state_dict(), 'checkpoint/headmodel.pt')
                    torch.save(tail_model.state_dict(), 'checkpoint/tailmodel.pt')
                    torch.save(ctx_proc.state_dict(), 'checkpoint/ctxmodel.pt')
                    torch.save(head_optimizer.state_dict(), 'checkpoint/head_optim.pt')
                    torch.save(tail_optimizer.state_dict(), 'checkpoint/tail_optim.pt')

if __name__ == '__main__':
    #print('Detecting NaN anomalies')
    #with torch.autograd.set_detect_anomaly(True):
    configure('logdir/' + str(datetime.datetime.now()))
    print(args)
    # delete all .png images in sample folder
    [os.remove(f) for f in glob.glob('sample/*.png')]

    # create folder if it doesn't exist for video samples
    if args.video_path is not None and not os.path.exists(args.video_path):
        os.makedirs(args.video_path)

    # if video path specified, delete all video* files!!!!!! start from scratch each time
    if args.video_path is not None:
        [os.remove(f) for f in glob.glob(os.path.join(args.video_path, 'video*'))]

    # create training and validation sets
    ds = MomentsInTimeDataset('../data/momentsintime/training', max_examples=args.max_examples, seed=100, single_example=True, single_label='walking')
    #val_ds = ds.split(0.9) # grab 10% of examples as validation set
    n_labels = len(ds.labels)
    print('Number of labels: %s' % n_labels)

    # this model generates the first frame
    head_model = LabelGlow(n_labels, args.img_size, 3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu,
                      nc_ctx=3 if args.use_label else 0)  # do not initialize act norm if restoring from save!
    head_model.to(device)
    print('Head model # params: %s' % calc_num_trainable_params(head_model.parameters()))
    print('Label emb # params: %s' % calc_num_trainable_params([head_model.l_embs]))

    # number of channels for context to glow - we use a context processor to give frame info to tail glow
    nc_context_for_tail = 3 if args.use_state is False else 3 + 16

    # this model generates all subsequent frames, or "tail" frames
    tail_model = Glow(3, args.n_flow // 16, args.n_block, affine=args.affine, conv_lu=not args.no_lu,
                      nc_ctx=nc_context_for_tail)
    tail_model.to(device)
    print('Tail model # params: %s' % calc_num_trainable_params(tail_model.parameters()))

    # this model processes all previous frames as context for next-frame generation
    ctx_proc = ContextProcessor(3, 16, args.n_frames)
    ctx_proc.to(device)
    print('Context processor # params: %s' % calc_num_trainable_params(ctx_proc.parameters()))

    head_optimizer = optim.Adam(head_model.parameters(), lr=args.lr)
    tail_optimizer = optim.Adam(list(tail_model.parameters()) + list(ctx_proc.parameters()), lr=args.lr)

    if args.restore:
        print('Restoring from save!')
        restore_models_optimizers_from_save(head_model, tail_model, ctx_proc, head_optimizer, tail_optimizer)

    train(args, ds, head_model, tail_model, ctx_proc, head_optimizer, tail_optimizer,
          n_epochs=args.epochs, n_frames=args.n_frames)

    alert.write('glowtrain.py: Finished training')
