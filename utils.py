import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import random
import matplotlib.pyplot as plt


######################## Network initializers ################################
def netG(input_nc, output_nc, ngf, n_blocks, gpu_ids):
    """Create and initialize a generator"""
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False) # Instance norm
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=n_blocks)
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def netD(input_nc, ndf, n_layers, gpu_ids):
    """Create and initialize a discriminator"""
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False) # Instance norm
    net = Discriminator(input_nc, ndf, n_layers, norm_layer)
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


################# Weight initialization func #############################
def init_weights(net, init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain) # Normal weight initialization
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # apply the initialization function <init_func>


################### Generator networks ###############################
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9):
        super().__init__()
        use_bias = True

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
            model += [ResnetBlock(ngf * mult, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

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
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad2d(1)] # Reflection padding

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


############### Discriminator network ############################
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf, n_layers, norm_layer):
        super().__init__()
        use_bias = True

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
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
        return self.model(input)


########################### GAN Loss Module ############################
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss

################# Image Pool ##########################
class ImagePool():
    """Image buffer that stores previously generated images
    
    50/50 change to return either an input image or an image previously stored in the buffer
    """
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_idx = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_idx].clone()
                    self.images[random_idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


################## Learning rate scheduler ##################
def get_lr_scheduler(optimizer):
    """Returns a linear lr scheduler"""
    n_epochs = 100 # Number of epochs to keep initial learning rate
    n_epochs_decay = 100 # Decay lr over this many epochs
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch+1-n_epochs)/(n_epochs_decay+1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def save_loss_plot(losses, title):
    """Plots current losses and saves to image"""
    # Clear plot
    plt.clf()

    # Plot title
    plt.title(title)

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot losses
    plt.plot([x['G_A'] for x in losses], label='G_A')
    plt.plot([x['G_B'] for x in losses], label='G_B')
    plt.plot([x['D_A'] for x in losses], label='D_A')
    plt.plot([x['D_B'] for x in losses], label='D_B')

    # Generate legend
    plt.legend(loc="upper right")

    # Save to image
    plt.savefig("loss.png")
