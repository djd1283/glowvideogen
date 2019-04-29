"""Script to load and evaluate glow models"""
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from vgen.glowtrain import Glow, calc_loss
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
parser.add_argument('--n_frames', default=90, type=int, help='max number of video frames')
parser.add_argument('--max_examples', default=None, type=int, help='max number of examples to train on')

args = parser.parse_args()

head_model = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
head_model = head_model.to(device)
head_model.eval()

tail_model = Glow(3, args.n_flow // 32, args.n_block, affine=args.affine, conv_lu=not args.no_lu, nc_ctx=3)
tail_model = tail_model.to(device)
tail_model.eval()

# restore models from save
if args.restore:
    print('Restoring from save!')
    head_model.load_state_dict(torch.load(f'checkpoint/headmodel.pt'))
    tail_model.load_state_dict(torch.load(f'checkpoint/tailmodel.pt'))

ds = MomentsInTimeDataset('../data/momentsintime/training', max_examples=args.max_examples, seed=1212)
val_ds = ds.split(0.999) # grab .1% of examples as validation set
dl = DataLoader(val_ds, batch_size=args.batch, num_workers=3)
#print('Len validation dataset: %s' % len(val_ds))

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
        frames = frames - 0.5
        frames = torch.stack([downsample(frames[:, i]) for i in range(frames.shape[1])], 1)

        im = frames[:, 0]
        # calculate first frame loss
        log_p, logdet = head_model(im + torch.rand_like(im) / n_bins)
        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        # calculate loss for all frames
        losses = []
        for j in range(frames.shape[1] - 1):
            prev_frame = frames[:, j]
            frame = frames[:, j+1]
            delta = frame - prev_frame
            # calculate tail frame loss
            frame_log_p, frame_logdet = tail_model(delta + torch.rand_like(frame) / n_bins, ctx=prev_frame)
            frame_loss, frame_log_p, frame_log_det = calc_loss(frame_log_p, frame_logdet, args.img_size, n_bins)
            losses.append(frame_loss)

        total_loss = torch.stack(losses, dim=0).mean()
        frame_losses.append(total_loss)
        im_losses.append(loss)

    final_frame_loss = torch.stack(frame_losses, dim=0).mean()
    final_im_loss = torch.stack(im_losses, dim=0).mean()

print('First frame cross entropy: %s' % final_im_loss.item())
print('Tail frame cross entropy: %s' % final_frame_loss.item())