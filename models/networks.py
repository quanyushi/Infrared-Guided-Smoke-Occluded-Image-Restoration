import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'dual_branch_unet':
        # For dual branch, input_nc should be visible_nc + infrared_nc
        # We assume visible_nc = 3, infrared_nc = 3 by default
        visible_nc = 3
        infrared_nc = input_nc - visible_nc
        net = DualBranchUnetGenerator(visible_nc, infrared_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'dual_branch_discriminator':  # enhanced discriminator for dual branch generator
        net = DualBranchDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


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
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


##############################################################################
# Dual Branch UNet Generator for Infrared-Guided Image Reconstruction
##############################################################################

class DualBranchUnetGenerator(nn.Module):
    """
    Dual-branch UNet generator with:
    1. Visible light encoder for masked visible images
    2. Structure extraction encoder for infrared images
    3. Cross-modal structure guidance module
    4. Feature fusion decoder
    """

    def __init__(self, visible_nc, infrared_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Parameters:
            visible_nc (int)   -- number of channels in visible light images
            infrared_nc (int)  -- number of channels in infrared images
            output_nc (int)    -- number of channels in output images
            ngf (int)          -- number of filters in the last conv layer
            norm_layer         -- normalization layer
            use_dropout (bool) -- if use dropout layers
        """
        super(DualBranchUnetGenerator, self).__init__()

        # Visible light encoder (based on UNet encoder)
        self.visible_encoder = VisibleLightEncoder(visible_nc, ngf, norm_layer, use_dropout)

        # Structure extraction encoder for infrared
        self.structure_encoder = StructureExtractionEncoder(infrared_nc, ngf, norm_layer)

        # Cross-modal structure guidance module
        self.guidance_module = CrossModalGuidanceModule(ngf, norm_layer)

        # Feature fusion decoder
        self.fusion_decoder = FeatureFusionDecoder(ngf, output_nc, norm_layer, use_dropout)

    def forward(self, input_tensor):
        """
        Forward pass
        Args:
            input_tensor: concatenated input [B, visible_nc + infrared_nc, H, W]
                         First visible_nc channels: masked visible light image
                         Last infrared_nc channels: complete infrared image
        Returns:
            reconstructed visible image [B, output_nc, H, W]
        """
        # Split input into visible and infrared components
        visible_img = input_tensor[:, :3, :, :]  # First 3 channels for visible
        infrared_img = input_tensor[:, 3:, :, :]  # Remaining channels for infrared

        # Extract features from both branches
        vis_features = self.visible_encoder(visible_img)
        struct_features = self.structure_encoder(infrared_img)

        # Apply cross-modal guidance
        guided_features = self.guidance_module(vis_features, struct_features)

        # Decode to final output
        output = self.fusion_decoder(guided_features)

        return output


class VisibleLightEncoder(nn.Module):
    """
    Visible light encoder based on UNet encoder structure
    Extracts multi-scale features from masked visible images
    """

    def __init__(self, input_nc, ngf, norm_layer, use_dropout=False):
        super(VisibleLightEncoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Encoder layers (downsampling path)
        # Layer 1: input_nc -> ngf
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        )

        # Layer 2: ngf -> ngf*2
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, True)
        )

        # Layer 3: ngf*2 -> ngf*4
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.LeakyReLU(0.2, True)
        )

        # Layer 4: ngf*4 -> ngf*8
        self.down4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        # Layer 5: ngf*8 -> ngf*8
        self.down5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        # Layer 6: ngf*8 -> ngf*8
        self.down6 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        # Layer 7: ngf*8 -> ngf*8
        self.down7 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        # Bottleneck: ngf*8 -> ngf*8
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.ReLU(True)
        )

    def forward(self, x):
        """
        Forward pass through visible encoder
        Returns multi-scale features for skip connections
        """
        # Encoder forward pass with skip connections
        d1 = self.down1(x)      # [B, ngf, H/2, W/2]
        d2 = self.down2(d1)     # [B, ngf*2, H/4, W/4]
        d3 = self.down3(d2)     # [B, ngf*4, H/8, W/8]
        d4 = self.down4(d3)     # [B, ngf*8, H/16, W/16]
        d5 = self.down5(d4)     # [B, ngf*8, H/32, W/32]
        d6 = self.down6(d5)     # [B, ngf*8, H/64, W/64]
        d7 = self.down7(d6)     # [B, ngf*8, H/128, W/128]
        bottleneck = self.bottleneck(d7)  # [B, ngf*8, H/256, W/256]

        return {
            'skip1': d1,
            'skip2': d2,
            'skip3': d3,
            'skip4': d4,
            'skip5': d5,
            'skip6': d6,
            'skip7': d7,
            'bottleneck': bottleneck
        }


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient structure extraction
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, norm_layer=nn.BatchNorm2d):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=in_channels, bias=bias)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        # Normalization
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return x


class AsymmetricConvBlock(nn.Module):
    """
    Simplified asymmetric convolution block for capturing structural patterns
    """
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(AsymmetricConvBlock, self).__init__()

        # Simplified single path with asymmetric kernel
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))

        self.norm = norm_layer(out_channels)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        # Single asymmetric path
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.activation(out)

        return out


class StructureExtractionEncoder(nn.Module):
    """
    Structure extraction encoder for infrared images
    Uses depthwise separable convolutions and asymmetric receptive fields
    """

    def __init__(self, input_nc, ngf, norm_layer):
        super(StructureExtractionEncoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Simplified structure extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # Structure extraction layers using regular convolutions (more memory efficient)
        self.struct1 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf*2),
            nn.ReLU(True)
        )
        self.struct2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf*4),
            nn.ReLU(True)
        )
        self.struct3 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf*8),
            nn.ReLU(True)
        )

        self.activation = nn.ReLU(True)

    def forward(self, x):
        """
        Extract structural features from infrared image
        Returns edge features at multiple scales
        """
        # Initial feature extraction
        feat = self.initial_conv(x)  # [B, ngf, H, W]

        # Multi-scale structure extraction (simplified)
        struct1 = self.struct1(feat)      # [B, ngf*2, H/2, W/2]
        struct2 = self.struct2(struct1)   # [B, ngf*4, H/4, W/4]
        struct3 = self.struct3(struct2)   # [B, ngf*8, H/8, W/8]

        return {
            'edge0': feat,       # [B, ngf, H, W] - for skip1 guidance
            'edge1': struct1,    # [B, ngf*2, H/2, W/2]
            'edge2': struct2,    # [B, ngf*4, H/4, W/4]
            'edge3': struct3,    # [B, ngf*8, H/8, W/8]
        }


class SpatialAttentionGate(nn.Module):
    """
    Spatial attention gate for adaptive feature selection
    """
    def __init__(self, channels, norm_layer=nn.BatchNorm2d):
        super(SpatialAttentionGate, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, 1, kernel_size=1)
        self.norm = norm_layer(channels // 8)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # Generate attention map
        att = self.conv1(x)
        att = self.norm(att)
        att = self.relu(att)
        att = self.conv2(att)
        att = self.sigmoid(att)

        # Apply attention
        return x * att


class StructureGuidedGatingUnit(nn.Module):
    """
    Structure-guided gating unit for cross-modal feature fusion
    """
    def __init__(self, vis_channels, struct_channels, norm_layer=nn.BatchNorm2d):
        super(StructureGuidedGatingUnit, self).__init__()

        self.vis_channels = vis_channels
        self.struct_channels = struct_channels

        # Channel alignment if needed
        if vis_channels != struct_channels:
            self.channel_align = nn.Conv2d(struct_channels, vis_channels, kernel_size=1)
        else:
            self.channel_align = nn.Identity()

        # Structure-aware gating mechanism
        self.gate_conv = nn.Sequential(
            nn.Conv2d(vis_channels + vis_channels, vis_channels, kernel_size=3, padding=1),
            norm_layer(vis_channels),
            nn.ReLU(True),
            nn.Conv2d(vis_channels, vis_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(vis_channels, vis_channels, kernel_size=3, padding=1),
            norm_layer(vis_channels),
            nn.ReLU(True)
        )

        # Spatial attention for dynamic feature selection
        self.spatial_attention = SpatialAttentionGate(vis_channels, norm_layer)

    def forward(self, vis_feat, struct_feat):
        """
        Args:
            vis_feat: visible features [B, vis_channels, H, W]
            struct_feat: structure features [B, struct_channels, H, W]
        Returns:
            guided features [B, vis_channels, H, W]
        """
        # Align structure features to visible feature dimensions
        struct_aligned = self.channel_align(struct_feat)

        # Resize structure features to match visible features if needed
        if struct_aligned.shape[2:] != vis_feat.shape[2:]:
            struct_aligned = F.interpolate(struct_aligned, size=vis_feat.shape[2:],
                                         mode='bilinear', align_corners=False)

        # Generate structure-guided gate
        gate_input = torch.cat([vis_feat, struct_aligned], dim=1)
        gate = self.gate_conv(gate_input)

        # Apply gating to visible features
        gated_feat = vis_feat * gate + struct_aligned * (1 - gate)

        # Feature refinement
        refined_feat = self.refine_conv(gated_feat)

        # Apply spatial attention for dynamic feature selection
        output = self.spatial_attention(refined_feat)

        return output


class CrossModalGuidanceModule(nn.Module):
    """
    Cross-modal structure guidance module
    Implements spatial adaptive gating fusion and dynamic feature selection
    """

    def __init__(self, ngf, norm_layer=nn.BatchNorm2d):
        super(CrossModalGuidanceModule, self).__init__()

        # Gating units for different scales
        self.gate_skip1 = StructureGuidedGatingUnit(ngf, ngf, norm_layer)      # H/2 level
        self.gate_skip2 = StructureGuidedGatingUnit(ngf*2, ngf*2, norm_layer)  # H/4 level
        self.gate_skip3 = StructureGuidedGatingUnit(ngf*4, ngf*4, norm_layer)  # H/8 level
        self.gate_skip4 = StructureGuidedGatingUnit(ngf*8, ngf*8, norm_layer)  # H/16 level

    def forward(self, vis_features, struct_features):
        """
        Apply cross-modal guidance to visible features
        Args:
            vis_features: dict of visible features from encoder
            struct_features: dict of structure features from infrared encoder
        Returns:
            dict of guided features
        """
        guided_features = vis_features.copy()

        # Apply structure guidance at multiple scales
        # Match the scales between visible and structure features
        guided_features['skip1'] = self.gate_skip1(vis_features['skip1'], struct_features['edge0'])
        guided_features['skip2'] = self.gate_skip2(vis_features['skip2'], struct_features['edge1'])
        guided_features['skip3'] = self.gate_skip3(vis_features['skip3'], struct_features['edge2'])
        guided_features['skip4'] = self.gate_skip4(vis_features['skip4'], struct_features['edge3'])

        return guided_features


class FeatureFusionDecoder(nn.Module):
    """
    Feature fusion decoder that reconstructs the final output
    Uses guided features with skip connections
    """

    def __init__(self, ngf, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureFusionDecoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Decoder layers (upsampling path)
        # Up1: ngf*8 -> ngf*8 (with skip7)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        if use_dropout:
            self.up1.add_module('dropout', nn.Dropout(0.5))

        # Up2: ngf*8*2 -> ngf*8 (with skip6)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        if use_dropout:
            self.up2.add_module('dropout', nn.Dropout(0.5))

        # Up3: ngf*8*2 -> ngf*8 (with skip5)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        if use_dropout:
            self.up3.add_module('dropout', nn.Dropout(0.5))

        # Up4: ngf*8*2 -> ngf*4 (with guided skip4)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        # Up5: (ngf*4 + ngf*8) -> ngf*2 (up4 output + guided skip4)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 + ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        # Up6: (ngf*2 + ngf*4) -> ngf (up5 output + guided skip3)
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 + ngf * 4, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # Up7: (ngf + ngf*2) -> ngf (up6 output + guided skip2)
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(ngf + ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # Up8: (ngf + ngf) -> output_nc (up7 output + guided skip1) - Final layer
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(ngf + ngf, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, guided_features):
        """
        Decode guided features to final output
        Args:
            guided_features: dict containing guided features and skip connections
        Returns:
            final reconstructed image [B, output_nc, H, W]
        """
        # Start from bottleneck
        x = guided_features['bottleneck']

        # Decoder with skip connections
        x = self.up1(x)  # [B, ngf*8, H/128, W/128]
        x = torch.cat([x, guided_features['skip7']], dim=1)  # Concatenate skip7

        x = self.up2(x)  # [B, ngf*8, H/64, W/64]
        x = torch.cat([x, guided_features['skip6']], dim=1)  # Concatenate skip6

        x = self.up3(x)  # [B, ngf*8, H/32, W/32]
        x = torch.cat([x, guided_features['skip5']], dim=1)  # Concatenate skip5

        x = self.up4(x)  # [B, ngf*4, H/16, W/16]
        x = torch.cat([x, guided_features['skip4']], dim=1)  # Concatenate guided skip4

        x = self.up5(x)  # [B, ngf*2, H/8, W/8]
        x = torch.cat([x, guided_features['skip3']], dim=1)  # Concatenate guided skip3

        x = self.up6(x)  # [B, ngf, H/4, W/4]
        x = torch.cat([x, guided_features['skip2']], dim=1)  # Concatenate guided skip2

        x = self.up7(x)  # [B, ngf, H/2, W/2]
        x = torch.cat([x, guided_features['skip1']], dim=1)  # Concatenate guided skip1

        x = self.up8(x)  # [B, output_nc, H, W] - Full resolution output

        return x


class DualBranchDiscriminator(nn.Module):
    """
    Enhanced discriminator for dual-branch generator
    Uses deeper architecture with attention mechanism
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        Parameters:
            input_nc (int) -- number of input channels (masked_image + generated_image)
            ndf (int) -- number of filters in the first conv layer
            norm_layer -- normalization layer
        """
        super(DualBranchDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Enhanced single-scale discriminator with more layers
        self.main_disc = self._make_discriminator(input_nc, ndf, norm_layer, use_bias, n_layers=4)

        # Spatial attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(input_nc, ndf//4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ndf//4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_discriminator(self, input_nc, ndf, norm_layer, use_bias, n_layers=3):
        """Create a single-scale discriminator"""
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        return nn.Sequential(*sequence)

    def forward(self, input):
        """
        Enhanced single-scale discrimination with attention
        """
        # Generate attention map
        attention_map = self.attention(input)

        # Apply attention to input
        attended_input = input * attention_map

        # Main discrimination
        output = self.main_disc(attended_input)

        return output
