import os
import skvideo
import skvideo.io
import numpy as np
import skvideo.datasets

# def downloadYouTube(videourl, path):
#
#     yt = YouTube(videourl)
#     yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
#     if not os.path.exists(path):
#         os.makedirs(path)
#     yt.download(path)
#
# downloadYouTube('https://www.youtube.com/watch?v=zNyYDHCg06c', 'video')

sample_video = '../data/Moments_in_Time_Mini/training/cheering/yt-0gfPFL9K62o_42.mp4'
#videodata = skvideo.io.vread(sample_video)

# videogen = skvideo.io.vreader(sample_video)
# frames = []
# for frame in videogen:
#     frames.append(frame)

#reader = skvideo.io.FFmpegReader(sample_video, inputdict={'-r': '30'})

# iterate through the frames

inputparameters = {}
outputparameters = {}
bunnyname = skvideo.datasets.bigbuckbunny()
print(bunnyname)
reader = skvideo.io.FFmpegReader(sample_video,
                inputdict=inputparameters,
                outputdict=outputparameters)

frames = []
for frame in reader.nextFrame():
    frames.append(frame)

videodata = np.stack(frames, axis=0)

print(np.min(videodata))
print(np.max(videodata))
print(np.mean(videodata))

print(videodata.shape)
print(videodata)