import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True, initialized = False):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.initialized = initialized
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if not self.initialized:
            self.initialize(input)
            self.initialized = True

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True, nc_ctx=0):
        super().__init__()
        self.nc_ctx = nc_ctx
        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2 + nc_ctx, filter_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input, ctx=None):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            if ctx is None:
                log_s, t = self.net(in_a).chunk(2, 1)
            else:
                assert self.nc_ctx > 0
                # add context to input of affine coupling - conditional glow!
                log_s, t = self.net(torch.cat([in_a, ctx], 1)).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            if ctx is None:
                net_out = self.net(in_a)
            else:
                assert self.nc_ctx > 0
                # add context to input of affine coupling - conditional glow!
                net_out = self.net(torch.cat([in_a, ctx], 1))
            #net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, ctx=None):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            if ctx is None:
                log_s, t = self.net(out_a).chunk(2, 1)
            else:
                assert self.nc_ctx > 0
                # add context to input of affine coupling - conditional glow!
                log_s, t = self.net(torch.cat([out_a, ctx], -1))
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            if ctx is None:
                net_out = self.net(out_a)
            else:
                assert self.nc_ctx > 0
                # add context to input of affine coupling - conditional glow!
                net_out = self.net(torch.cat([out_a, ctx], 1))
            # net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True, nc_ctx=0, act_norm_init=False):
        super().__init__()

        self.actnorm = ActNorm(in_channel, initialized=act_norm_init)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine, nc_ctx=nc_ctx)

    def forward(self, input, ctx=None):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out, ctx=ctx)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output, ctx=None):
        input = self.coupling.reverse(output, ctx=ctx)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    # ADJUSTED LOG P BY INVERTING EXP TO MAKE IT MORE STABLE
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 * torch.exp(-2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

def squeeze_image(input):
    """Half the width and height of the image, but expand the number of channels by 4."""
    b_size, n_channel, height, width = input.shape
    squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
    squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
    out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
    return out

def unsqueeze_image(input):
    """Reduce the number of channels by 4, but double the width and height of the image. Inverse operation
    of squeeze_image()."""
    b_size, n_channel, height, width = input.shape
    unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
    unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
    unsqueezed = unsqueezed.contiguous().view(
        b_size, n_channel // 4, height * 2, width * 2
    )
    return unsqueezed

class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, nc_ctx=0, act_norm_init=False):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu, nc_ctx=nc_ctx, act_norm_init=act_norm_init))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input, ctx=None):
        b_size = input.shape[0]
        out = squeeze_image(input)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out, ctx=ctx)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p

    def reverse(self, output, eps=None, ctx=None):
        input = output

        if self.split:
            mean, log_sd = self.prior(input).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = torch.cat([output, z], 1)

        else:
            zero = torch.zeros_like(input)
            # zero = F.pad(zero, [1, 1, 1, 1], value=1)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input, ctx=ctx)

        unsqueezed = unsqueeze_image(input)

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, nc_ctx=0, act_norm_init=False):
        """

        :param in_channel:
        :param n_flow:
        :param n_block:
        :param affine:
        :param conv_lu:
        :param nc_ctx:
        :param act_norm_init: actnorm is initialized - if False, initialize it from first batch (True for restore)
        """
        super().__init__()
        if nc_ctx > 0:
            self.ctx_downs = nn.ModuleList(
                [nn.Conv2d(nc_ctx * 4, nc_ctx, 3, stride=1, padding=2, dilation=2)
                 for i in range(n_block-1)])
            self.relu = nn.ReLU()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, nc_ctx=nc_ctx * 4, act_norm_init=act_norm_init))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine, nc_ctx=nc_ctx * 4, act_norm_init=act_norm_init))

    def build_ctx_pyramid(self, ctx):
        # construct a pyramid from input context
        ctx_pyramid = None
        if ctx is not None:
            ctx_pyramid = [squeeze_image(ctx)]  # nc_ctx * 4
            for down in self.ctx_downs:
                layer = down(ctx_pyramid[-1])  # nc_ctx * 2
                layer = squeeze_image(layer)  # nc_ctx * 8
                layer = self.relu(layer)  # activation not applied to input image
                ctx_pyramid.append(layer)  # ctx_pyramid[-1] -> nc_ctx * 8
        return ctx_pyramid

    def forward(self, input, ctx=None):
        log_p_sum = 0
        logdet = 0
        out = input

        if ctx is not None:
            ctx_pyramid = self.build_ctx_pyramid(ctx)

        for i, block in enumerate(self.blocks):
            ctx_in = None if ctx is None else ctx_pyramid[i]
            out, det, log_p = block(out, ctx=ctx_in)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet

    def reverse(self, z_list, ctx=None):

        if ctx is not None:
            ctx_pyramid = self.build_ctx_pyramid(ctx)
            ctx_pyramid.reverse()  # we are introducing pyramid in reverse order now

        for i, block in enumerate(self.blocks[::-1]):
            ctx_in = None if ctx is None else ctx_pyramid[i]
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], ctx=ctx_in)

            else:
                input = block.reverse(input, z_list[-(i + 1)], ctx=ctx_in)

        return input


class ContextProcessor(nn.Module):
    def __init__(self, nc_frame, nc_state):
        """Special module that reads frame by frame to capture context. This context is then fed
        into the glow at each step to incorporate context."""
        super().__init__()
        self.nc_state = nc_state
        self.conv1 = nn.Conv2d(nc_frame + nc_state, nc_state * 2, 3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(nc_state * 2, nc_state, 3, stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()

    def forward(self, frame, state):
        """Process frame and state into new state."""
        both = torch.cat([frame, state], dim=1)  # cat along channel dimension
        delta = self.conv2(self.relu(self.conv1(both)))
        return state + delta













