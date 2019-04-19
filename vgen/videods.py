"""Module containing torch Dataset loader for Moments in Time video dataset."""
import skvideo.io
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from tqdm import tqdm


class MomentsInTimeDataset(Dataset):
    def __init__(self, data_dir, max_timesteps=90, width=256, height=256, channels=3, max_examples=None,
                 single_example=False, transform=None):
        """Dataset class loads the Moments in Time dataset, and dispenses (label, frames)
        examples."""
        super().__init__()
        self.classes = []
        self.data_dir = data_dir
        self.max_timesteps = max_timesteps
        self.labels = {}
        self.width = width
        self.height = height
        self.channels = channels
        self.max_examples = max_examples
        self.single_example=single_example
        self.transform = transform

        self.examples = []
        # load dataset
        for dirName, subdirList, fileList in os.walk(data_dir):
            # go through all label directories
            # skip parent data directory
            # parent folder becomes name of class
            label = os.path.basename(dirName)
            self.labels[label] = len(self.labels) # O(1) mapping from label to index
            for file in fileList:
                # for each .mp4 file, add to list of examples
                if file.endswith('.mp4'):
                    self.examples.append((label, os.path.join(dirName, file)))
        # randomize
        shuffle(self.examples)

        self.single_frames, self.single_label = None, None
        if single_example:
            label, file = self.examples[0]
            print('filename: %s' % file)
            self.single_label = np.array(self.labels[label], dtype=np.int)
            self.single_frames = self.load_frames(file)

    def __len__(self):
        if self.max_examples is None:
            return len(self.examples)
        else:
            return self.max_examples

    def load_frames(self, file):
        """Load frames of file from path.

        :return Numpy array (N, C, H, W)"""
        frames = load_video(file)
        frames = frames[:self.max_timesteps]  # discard extra frames
        num_frames = frames.shape[0]
        diff = self.max_timesteps - num_frames
        if diff > 0:
            pad = np.zeros([diff, self.width, self.height, self.channels], dtype=np.uint8)
            frames = np.concatenate([frames, pad], axis=0)

        # swap axes from TCWH to THWC to confirm with pytorch formatting
        frames = np.swapaxes(frames, 1, 3)

        # if transform provided, use it on every frame
        transformed_frames = []
        if self.transform is not None:
            for i in range(frames.shape[0]):
                transformed_frames.append(self.transform(frames[i]))
        frames = np.stack(transformed_frames, 0)

        # convert frame to normalized float
        frames = (frames / 256.0).astype(np.float32)

        return frames

    def __getitem__(self, idx):
        """Dispense example containing the frames of a video and the class it belongs to.

        :return returns tuple (label, frames) where label is zero-dim nd.array containing index of
        class it belongs to. frames is nd.array of shape (num_frames, width, height, 3) for RGB values."""
        if not self.single_example:
            label, file = self.examples[idx]
            label_idx = np.array(self.labels[label], dtype=np.int)
            frames = self.load_frames(file)
        else:
            label_idx, frames = self.single_label, self.single_frames

        return label_idx, frames


def load_video(filename):
    """
    Read video file (e.g. mp4) from disk an return numpy array.

    :param filename: string indicating location of file
    :return: nd.array shape (num_frames, width, height, 3) for RGB values of all frames
    """
    reader = skvideo.io.FFmpegReader(filename,
                                     inputdict={},
                                     outputdict={})
    frames = []
    for frame in reader.nextFrame():
        frames.append(frame)

    return np.stack(frames, axis=0)

if __name__ == '__main__':
    ds = MomentsInTimeDataset('../data/momentsintime/training')
    print(ds.labels)
    print('Num examples: %s' % len(ds.examples))
    label, frames = ds[0]
    print('Label shape: %s' % str(label.shape))
    print('Frames shape: %s' % str(frames.shape))
    print('Frames mean: %s' % np.mean(frames))
    print('Frames std: %s' % np.std(frames))

    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=5)
    for batch in tqdm(dl):
        label, frames = batch

