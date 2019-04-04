"""Here we have video generation models and other layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class VideoGeneratorBaseline(nn.Module):
    def __init__(self, n_frames, nc_z):
        """This generator takes in a random vector and converts it into a video. Can also take
        contextual information."""
        super().__init__()
        self.im_gen = ImageGeneratorBaseline(nc_z, 6)
        self.unet = UnetGenerator(6, 3)
        self.out = nn.Conv2d(9, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.n_frames = n_frames

    def forward(self, z):
        """Take input z and produce output frames."""
        two_init_frames = self.im_gen(z)
        first_frame = two_init_frames[:, :3, ...]
        delta = two_init_frames[:, 3:, ...]  # two initial frames needed for motion
        frames = [first_frame, first_frame + delta]
        for i in range(self.n_frames - 2):
            prev_frame = torch.cat(frames[-2:], 1)
            assert prev_frame.shape[1] == 6  # two frames
            next_frame_delta = self.unet(prev_frame)
            frame_and_delta = torch.cat(frames[-2:] + [next_frame_delta], 1)
            next_frame = self.out(frame_and_delta)
            frames.append(frames[-1] + next_frame)

        assert len(frames) == self.n_frames
        video = torch.stack(frames, 1)
        # NO TANH OUTPUT AT THE MOMENT
        return torch.tanh(video) # b x t x c x w x h


class VideoDiscriminatorBaseline(nn.Module):
    def __init__(self, nc_extra=3):
        super().__init__()
        self.nc_extra = nc_extra
        self.unet = UnetGenerator(3 + nc_extra, nc_extra, use_dropout=True)
        self.predictor = nn.Conv2d(8, 1, kernel_size=1,
                             stride=2, padding=1, bias=True)

    def forward(self, x):
        """Take video frames and produce a prediction as to whether
        the video is real or generated.

        x: torch tensor shape [batch_size, num_frames, channels, width, height]"""
        b, n_frames, nc, width, height = x.shape
        video_data = torch.zeros(x.shape[0], self.nc_extra, width, height).to(x.device)
        scores = []
        for i in range(n_frames):
            unet_in = torch.cat([x[:,i], video_data], 1)
            video_data = self.unet(unet_in)

            highest_layer = self.unet.outputs[-1]
            prediction = self.predictor(highest_layer)
            # we sum across entire prediction map to get single scalar score
            score = torch.mean(torch.mean(prediction, dim=-1), dim=-1).squeeze(-1) # b
            scores.append(score)

        return torch.stack(scores, 1)


class ImageGeneratorBaseline(nn.Module):
    def __init__(self, nc_in, nc_out):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(nc_in, 4,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1, bias=True)
        self.relu1 = nn.Tanh()

        self.conv2 = nn.ConvTranspose2d(4, 8,
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,bias=True)
        self.relu2 = nn.Tanh()

        self.conv3 = nn.ConvTranspose2d(8, 16,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=True)

        self.relu3 = nn.Tanh()

        self.conv4 = nn.ConvTranspose2d(16, nc_out,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=True)

        self.model = [self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3, self.conv4]

    def forward(self, z):
        for i, comp in enumerate(self.model):
            z = comp(z)
            if i != len(self.model) - 1: z = z * 2
        return z


"""Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py"""
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs=5, ngf=8, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.outputs = None
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf, ngf, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf, ngf, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        out = self.model(input)
        self.outputs = self.model.outputs  # grab list of outputs from
        return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.submodule = submodule
        self.outputs = None

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, bias=use_bias, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # evaluate output of unet block
        #print('Input: %s' % str(x.shape))
        out = self.model(x)
        #print('Output: %s' % str(out.shape))
        # extra block of code to grab list of outputs
        if self.submodule is not None:
            self.outputs = self.submodule.outputs
        else:
            self.outputs = []
        self.outputs = [out] + self.outputs

        if self.outermost:
            return out
        else:  # add skip connections
            return torch.cat([x, out], 1)