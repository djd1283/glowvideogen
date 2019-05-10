"""Script to load and evaluate glow models"""
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from vgen.glowtrain import calc_loss, calc_video_loss
from vgen.glowmodel import Glow, LabelGlow, ContextProcessor
from vgen.videods import MomentsInTimeDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=16, type=int, help='batch size')
parser.add_argument('--n_flow', default=32, type=int, help='number of flows in each block')
parser.add_argument('--restore', default=False, action='store_true', help='if true, restore from saved checkpoint')
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', action='store_true', help='use affine coupling instead of additive')
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
parser.add_argument('--path', metavar='PATH', type=str, help='Path to image directory')
parser.add_argument('--video_path', metavar='PATH', type=str, help='Where to save generated video samples')
parser.add_argument('--n_frames', default=30, type=int, help='max number of video frames')
parser.add_argument('--max_examples', default=None, type=int, help='max number of examples to train on')
parser.add_argument('--use_state', default=False, action='store_true', help='if true, allow state info to pass between frames')
parser.add_argument('--use_label', default=False, action='store_true', help='if true, allow state info to pass between frames')
parser.add_argument('--train', default=False, action='store_true', help='if true, evaluate on training examples')

args = parser.parse_args()

# we pick a seed of 1212 arbitrarily, but match it with the train script to ensure even split
ds = MomentsInTimeDataset('../data/momentsintime/training', max_examples=args.max_examples, seed=100)

n_labels = len(ds.labels)

if args.train:
    val_ds = ds.split(0.001) # grab .1% of examples as validation set
    dl = DataLoader(ds, batch_size=args.batch, num_workers=3)
else:
    # evaluate on validation examples
    val_ds = ds.split(0.999)
    dl = DataLoader(val_ds, batch_size=args.batch, num_workers=3)

print('Number of batches to evaluate: %s' % len(dl))

head_model = LabelGlow(n_labels, args.img_size, 3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu,
                nc_ctx=3 if args.use_label else 0)
head_model.to(device)
head_model.train()

# number of channels for context to glow - we use a context processor to give frame info to tail glow
nc_context_for_tail = 3 if args.use_state is False else 3 + 16

# this model generates all subsequent frames, or "tail" frames
tail_model = Glow(3, args.n_flow // 32, args.n_block, affine=args.affine, conv_lu=not args.no_lu,
                  nc_ctx=nc_context_for_tail)
tail_model.to(device)
tail_model.train()

# this model processes all previous frames as context for next-frame generation
ctx_proc = ContextProcessor(3, 16, args.n_frames)
ctx_proc.to(device)
ctx_proc.train()

# restore models from save
if args.restore:
    print('Restoring from save!')
    head_model.load_state_dict(torch.load(f'checkpoint/headmodel.pt'))
    tail_model.load_state_dict(torch.load(f'checkpoint/tailmodel.pt'))
    ctx_proc.load_state_dict(torch.load(f'checkpoint/ctxmodel.pt'))

down_x = 256 // args.img_size
downsample = nn.AvgPool2d(down_x, stride=down_x, padding=0).to(device)
n_bins = 2. ** args.n_bits

pbar = tqdm(dl)

frame_losses = []
im_losses = []  # first frame loss

with torch.no_grad():
    for batch_idx, batch in enumerate(pbar):
        batch = [t.to(device) for t in batch]

        # preprocess frames
        label, frames = batch
        label = label if args.use_label else None
        frames = frames[:, :args.n_frames]
        frames = frames - 0.5
        frames = torch.stack([downsample(frames[:, i]) for i in range(frames.shape[1])], 1)
        assert frames.shape[1] == 30
        im = frames[:, 0]
        # calculate first frame loss
        log_p, logdet = head_model(im + torch.rand_like(im) / n_bins, label=label)
        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        # calculate loss for all frames
        total_loss = calc_video_loss(tail_model, ctx_proc, frames, n_bins)
        frame_losses.append(total_loss)
        im_losses.append(loss)

    final_frame_loss = torch.stack(frame_losses, dim=0).mean()
    final_im_loss = torch.stack(im_losses, dim=0).mean()

print('First frame cross entropy: %s' % final_im_loss.item())
print('Tail frame cross entropy: %s' % final_frame_loss.item())