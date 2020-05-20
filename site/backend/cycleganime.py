import torch
import torch.nn as nn
import functools

class CycleGANime():
    """CycleGAN network (just the generator A) for serving"""
    def __init__(self, n_blocks, ngf):
        self.device = torch.device('cpu')
        self.netG_A = ResnetGenerator(input_nc=3, output_nc=3, ngf=ngf, n_blocks=n_blocks).to(self.device)
        self.netG_A.eval()

    def run_inference(self, im):
        with torch.no_grad():
            im = im.to(self.device)
            return self.netG_A(im.unsqueeze(0)).permute(0,2,3,1).numpy()

    def load_weights(self, weights_path):
        """Load latest weights to resume training"""
        netG_A_state_dict = torch.load(weights_path, map_location=self.device)
        self.netG_A.load_state_dict(netG_A_state_dict)


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
