import torch
import torch.nn as nn
from torch.nn import init
import functools

######################## Network initializers ################################
def define_netG(input_nc, output_nc, ngf, n_blocks, device):
    """Create and initialize a generator"""
    net = ResnetGenerator(input_nc, output_nc, ngf, n_blocks).to(device)
    init_weights(net)
    return net

def define_netD(input_nc, ndf, n_layers, device):
    """Create and initialize a discriminator"""
    net = Discriminator(input_nc, ndf, n_layers).to(device)
    init_weights(net)
    return net
    

################# Weight initialization func #############################
def init_weights(net, init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, init_gain) # Normal weight initialization
            init.constant_(m.bias.data, 0.0) # All our conv layers have bias
    net.apply(init_func)  # apply initialization function


################### Generator network ###############################
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, n_blocks):
        super().__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = []

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            # 2 downsampling layers
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            norm_layer(ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        ]

        # Add resnet blocks
        for i in range(n_blocks): 
            model += [ResnetBlock(ngf * 4, norm_layer=norm_layer)]
        
        model += [
            # Add 2 upsampling layers (same amount as downsampling layers)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            norm_layer(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            # Back where we started
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer)

    def build_conv_block(self, dim, norm_layer):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True), 
            norm_layer(dim), 
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True), 
            norm_layer(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # residual connections
        return out


############### Discriminator network ############################
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf, n_layers):
        super().__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = []

        model += [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


########################### GAN Loss Module ############################
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss
