# glowvideogen
Conditional video generation project with the Glow model.

The Glow model in this repository was adapted from the Glow implementation of https://github.com/rosinality/glow-pytorch.
Thank you for the starter code!

How to use:

Download the repo, add the root glowvideogen to the PYTHONPATH variable in .bashrc (or .profile).

Download the MIT Moments in Time dataset from this webpage: http://moments.csail.mit.edu/ and place it in the same directory
as the vgen code directory.

Run python glowtrain.py to train the Glow model. During training, first frame generation samples will appear in the sample folder
in vgen, while generated videos will appear in the video_sample directory in vgen. A sample run would look like:

```
python glowtrain.py --img_size=64 --batch=8 --video_path=video_sample --use_state --use_label --epochs=1 --n_frames=30
```

While the evaluation script can be involked as:

```
python gloweval.py --img_size=64 --batch=8 --restore --use_label --use_state
```

The demo script demo.ipynb should be used to print sample videos and images, and while code exists to generate GAN videos,
this code is no longer in use.
