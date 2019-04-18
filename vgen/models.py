"""Here we have video generation models and other layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class VideoGeneratorBaseline(nn.Module):
    def __init__(self, nc_z):
        """This generator takes in a random vector and converts it into a video. Can also take
        contextual information."""
        super().__init__()
        self.unet = UnetGenerator(3, 8, use_input=True)
        self.predictor = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.gen_image = nn.Parameter(torch.randn(1, 1, 3, 16, 16), requires_grad=True)

    def forward(self, out_pyramid, n_frames=90):
        """Take input z and produce output frames."""
        batch_size = out_pyramid[0].shape[0]
        frame = self.predictor(out_pyramid[0])
        frames = [frame]
        for i in range(n_frames-1):
            out_pyramid = self.unet(x=frames[-1], in_pyramid=out_pyramid)  # give pyramid and previous frame
            frame = self.predictor(out_pyramid[0])
            frames.append(frame)

        video = torch.stack(frames, 1)
        # CHANGED OUTPUT OF THIS FUNCTION TO RETURN LEARNED BIAS FRAME
        bias_frame = self.gen_image.expand(batch_size, n_frames, -1, -1, -1)
        return bias_frame, out_pyramid # b x t x c x w x h


class VideoDiscriminatorBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_frame_unet = UnetGenerator(3, 3, use_input=True, use_pyramid=False, skip_connect=False, use_dropout=True)
        self.unet = UnetGenerator(3, 3, use_input=True, use_dropout=True)
        self.predictor = nn.Conv2d(8 * 8, 1, kernel_size=1, stride=2, padding=1, bias=False)
        self.linear_predictor = nn.Linear(16 * 16 * 3, 1)

    def forward(self, x):
        """Take video frames and produce a prediction as to whether
        the video is real or generated.

        x: torch tensor shape [batch_size, num_frames, channels, width, height]"""
        b, n_frames, nc, width, height = x.shape
        # scores = []
        #
        #   # batched first frame
        #
        # output_pyramid = None
        # for i in range(n_frames):
        #     if i == 0:
        #         output_pyramid = self.first_frame_unet(x=x[:, 0])  # this one doesn't require input pyramid
        #     else:
        #         output_pyramid = self.unet(x=x[:, i], in_pyramid=output_pyramid)
        #
        #     highest_layer = output_pyramid[-1]
        #     prediction = self.predictor(highest_layer)
        #     # we sum across entire prediction map to get single scalar score
        #     score = prediction.mean(-1).mean(-1).squeeze(-1) # b
        #     scores.append(score)

        # CHANGING DISCRIMINATOR TO SIMPLE LINEAR CLASSIFIER
        x_flat = x.reshape(b, n_frames, -1)
        scores = self.linear_predictor(x_flat)
        return scores.mean(1)

        #return torch.stack(scores, 1).mean(1)


class ImageGeneratorBaseline(nn.Module):
    def __init__(self, nc_in, nc_out, num_layers=10):
        """Generates the initial image at different levels of resolution."""
        super().__init__()
        # produce abstract representations of first frame in video
        # no skip connections, each unet performs entirely different transformation
        self.unet1 = UnetGenerator(nc_in, nc_out, skip_connect=False, use_input=True, use_pyramid=False)
        unets = []
        for i in range(num_layers - 1):
            unet = UnetGenerator(nc_in, nc_out, skip_connect=True, use_input=False)
            unets.append(unet)

        self.model = nn.ModuleList(unets)

    def forward(self, z):
        out = self.unet1(z)
        for unet in self.model:
            out = unet(in_pyramid=out)
        return out


"""Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py"""
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs=4, ngf=8, norm_layer=nn.BatchNorm2d, skip_connect=True,
                 use_input=False, use_pyramid=True, use_dropout=False):
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
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None, submodule=None, norm_layer=norm_layer, use_dropout=use_dropout,
                                             innermost=True, skip_connect=skip_connect, use_pyramid=use_pyramid)  # add the innermost layer
        for i in range(num_downs - 4):  # add intermediate layers with ngf filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout,
                                                 norm_layer=norm_layer, skip_connect=skip_connect, use_pyramid=use_pyramid)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout,
                                             norm_layer=norm_layer, skip_connect=skip_connect, use_pyramid=use_pyramid)
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 4, input_nc=None, submodule=unet_block, use_dropout=use_dropout,
        #                                      norm_layer=norm_layer, skip_connect=skip_connect, use_pyramid=use_pyramid)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             skip_connect=skip_connect, use_pyramid=use_pyramid, use_dropout=use_dropout)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, skip_connect=skip_connect, use_dropout=use_dropout,
                                             use_input=use_input, use_pyramid=use_pyramid)  # add the outermost layer

    def forward(self, x=None, in_pyramid=None):
        """Standard forward"""
        out = self.model(x=x, in_pyramid=in_pyramid)
        return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, skip_connect=True, use_input=True, use_pyramid=True, use_dropout=False):
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
        self.innermost = innermost
        self.submodule = submodule
        self.outputs = None
        self.use_input = use_input
        self.use_pyramid = use_pyramid
        self.skip_connect = skip_connect  # allow output pyramid to be built by adding to input pyramid

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        # we now take as input both input features and features from the previous pyramid
        if use_input and use_pyramid:
            in_size = input_nc + outer_nc
        elif use_input:
            in_size = input_nc
        elif use_pyramid:
            in_size = outer_nc
        else:
            raise AssertionError("Must use either input and/or pyramid features")

        downconv = nn.Conv2d(in_size, inner_nc, kernel_size=3, dilation=2,
                             stride=2, bias=use_bias, padding=2)
        downrelu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        downnorm = norm_layer(inner_nc)
        self.drop = nn.Dropout(0.7)
        self.uprelu = nn.ReLU(True)
        self.upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1)
            down = [downconv]
            up = [self.uprelu, upconv]
            #model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [self.uprelu, upconv]
            #model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [self.uprelu, upconv]

        if use_dropout:
            down = down + [self.drop]
            up = up + [self.drop]

        self.out = nn.Conv2d(in_size + outer_nc, outer_nc, kernel_size=3,
                             stride=1, bias=use_bias, padding=1)
        self.gate = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                             stride=1, bias=use_bias, padding=1)

        #self.model = nn.Sequential(*model)
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, x=None, in_pyramid=None):
        """
        Compute forward pass of U-Net operating on states.
        :param x input context, most-likely abstract features computed from less-abstract features (conv)
        :in_pyramid pyramid of feature representations for image at different levels of abstraction
        :return: Next frame output pyramid
        """
        assert x is not None or self.use_input is False
        assert in_pyramid is not None or self.use_pyramid is False
        assert x is None or self.use_input
        assert in_pyramid is None or self.use_pyramid

        # here we break apart pyramid and remove the bottom layer, or build new pyramid
        if self.use_pyramid:
            state = in_pyramid[0]
            rest_of_pyramid = in_pyramid[1:]
            # WARNING: Having a different sized x_and_layer could cause filter size issues!
            if self.use_input:
                x_and_layer = torch.cat([x, state], 1)
            else:
                x_and_layer = state  # allow user to not have any input context x
        else:
            # allow option of no initial pyramid
            state = None
            rest_of_pyramid = None
            x_and_layer = x

        # here we produce layer of pyramid, and use submodule to build the rest of the pyramid upward for us
        next_x = self.down(x_and_layer)
        if self.submodule is not None:
            out_pyramid = self.submodule(x=next_x, in_pyramid=rest_of_pyramid)
            up_layer = self.uprelu(self.up(out_pyramid[0]))  # bottom/middle of pyramid
        else:
            out_pyramid = []
            up_layer = self.uprelu(self.up(next_x))  # top of pyramid

        x_and_layer_and_up = torch.cat([x_and_layer, up_layer], 1)
        next_layer = self.out(x_and_layer_and_up)  # skip connection between previous

        #if self.skip_connect:
            # use update gate as in gru
            #update_gate = torch.sigmoid(self.gate(next_layer))
            #next_layer = update_gate * state + (1 - update_gate) * next_layer

        if self.skip_connect:
            next_layer = state + next_layer

        # normalize output
        # if not self.outermost and not self.innermost:
        #     next_layer = self.upnorm(next_layer)

        out_pyramid = [next_layer] + out_pyramid  # slip layer onto bottom of abstract state pyramid

        return out_pyramid


class MyLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        """Performs layer normalization over input.
        :param x Tensor shape (batch_size, n_channels, width, height)

        Returns: normalized input."""
        bs, nc, h, w = x.shape
        x_flat = x.view(bs, nc, -1)
        mean = x_flat.mean(-1).unsqueeze(-1).unsqueeze(-1) # take mean across width and height, bs x nc
        std = x_flat.std(-1).unsqueeze(-1).unsqueeze(-1)  # bs x nc
        return (x - mean) / (std + eps)  # normalize across each channel separately