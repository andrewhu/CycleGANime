import torch
import utils
import networks
import itertools
import numpy as np
import cv2
import random
import os
import glob

class CycleGANime():
    """Lineart colorization model based on CycleGAN
    
    This class is only for training the model
    """
    def __init__(self, input_nc=3, output_nc=3, gpu_id=None):
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else 'cpu')
        print(f"Using device {self.device}")

        # Hyperparameters
        self.lambda_idt = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0

        # Define generator networks
        self.netG_A = networks.define_netG(input_nc, output_nc, ngf=64, n_blocks=9, device=self.device)
        self.netG_B = networks.define_netG(output_nc, input_nc, ngf=64, n_blocks=9, device=self.device)

        # Define discriminator networks
        self.netD_A = networks.define_netD(output_nc, ndf=64, n_layers=3, device=self.device)
        self.netD_B = networks.define_netD(input_nc, ndf=64, n_layers=3, device=self.device)

        # Define image pools
        self.fake_A_pool = utils.ImagePool(pool_size=50)
        self.fake_B_pool = utils.ImagePool(pool_size=50)

        # Define loss functions
        self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # Define optimizers
        netG_params = itertools.chain(self.netG_A.parameters(), self.netG_B.parameters())
        netD_params = itertools.chain(self.netD_A.parameters(), self.netD_B.parameters())
        self.optimizer_G = torch.optim.Adam(netG_params, lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(netD_params, lr=0.0002, betas=(0.5, 0.999))

        # Learning rate schedulers
        self.scheduler_G = utils.get_lr_scheduler(self.optimizer_G)
        self.scheduler_D = utils.get_lr_scheduler(self.optimizer_D)

    def set_data(self, data):
        """Set input data"""
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)

    def forward(self):
        """Forward pass"""
        self.fake_B = self.netG_A(self.real_A)
        self.inv_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.inv_B = self.netG_A(self.fake_A)

    def backward_G(self):
        """Generator loss"""
        self.idt_A = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.lambda_B * self.lambda_idt

        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.lambda_A * self.lambda_idt

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        self.loss_cycle_A = self.criterionCycle(self.inv_A, self.real_A) * self.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.inv_B, self.real_B) * self.lambda_B

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def backward_D(self, netD, real, fake):
        """Calculates discriminator loss"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()
        
        return self.loss_D

    def optimize_parameters(self):
        # Compute forward pass
        self.forward()

        # Optimize generator (D needs no grads)
        for a, b in zip(self.netD_A.parameters(), self.netD_B.parameters()):
            a.requires_grad, b.requires_grad = False, False
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Optimize discriminator
        for a, b in zip(self.netD_A.parameters(), self.netD_B.parameters()):
            a.requires_grad, b.requires_grad = True, True
        self.optimizer_D.zero_grad()

        # Backwards D_A
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D(self.netD_A, self.real_B, fake_B)

        # Backwards D_B
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D(self.netD_B, self.real_A, fake_A)

        self.optimizer_D.step()

    def get_losses(self):
        return {
            'G_A': self.loss_G_A,
            'G_B': self.loss_G_B,
            'D_A': self.loss_D_A,
            'D_B': self.loss_D_B
        }

    def compute_visuals(self, test_data, sz, epoch, it):
        """Saves validation results"""
        with torch.no_grad():
            im = np.zeros((sz*8,sz*8,3))

            # Compute test results
            for idx, data in enumerate(test_data):
                self.set_data(data)
                self.forward()
                output_cpu = self.fake_B.permute(0,2,3,1).cpu().numpy()
                data_cpu = data['A'].permute(0,2,3,1).cpu().numpy()
                for i, img in enumerate(data_cpu):
                    im[idx*2*sz:idx*2*sz+sz, i*sz:i*sz+sz, :] = img
                for i, img in enumerate(output_cpu):
                    im[(idx*2+1)*sz:(idx*2+1)*sz+sz, i*sz:i*sz+sz, :] = img
            cv2.imwrite(f"results/result_{str(epoch).zfill(3)}_{str(it).zfill(4)}.jpg", cv2.cvtColor((im*255).astype(np.float32), cv2.COLOR_RGB2BGR))

    def save_weights(self, epoch, it):
        """Save model weights. Only save generator weights, except for the latest epoch"""

        checkpoints_base_folder = f"checkpoints"
        # Save generator weights
        folder_name = f"{checkpoints_base_folder}/{str(epoch).zfill(3)}_{str(it).zfill(4)}" # save folder
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        G_A_filename = f"{folder_name}/netG_A_{str(epoch).zfill(3)}_{str(it).zfill(4)}.pth"
        torch.save(self.netG_A.state_dict(), G_A_filename)

        # Save latest weights
        if not os.path.exists(f"{checkpoints_base_folder}/latest"):
            os.mkdir(f"{checkpoints_base_folder}/latest")
        G_A_filename = f"{checkpoints_base_folder}/latest/netG_A_latest.pth"
        G_B_filename = f"{checkpoints_base_folder}/latest/netG_B_latest.pth"
        D_A_filename = f"{checkpoints_base_folder}/latest/netD_A_latest.pth"
        D_B_filename = f"{checkpoints_base_folder}/latest/netD_B_latest.pth"
        torch.save(self.netG_A.state_dict(), G_A_filename)
        torch.save(self.netG_B.state_dict(), G_B_filename)
        torch.save(self.netD_A.state_dict(), D_A_filename)
        torch.save(self.netD_B.state_dict(), D_B_filename)

    def load_weights(self, epoch):
        """Load latest weights to resume training"""
        checkpoints_base_folder = f"checkpoints"


        weights_path = f"{checkpoints_base_folder}/latest"
        print(f"Loading weights from '{weights_path}'")

        netG_A_state_dict = torch.load(os.path.join(f"{checkpoints_base_folder}/latest/netG_A_latest.pth"), map_location=self.device)
        netG_B_state_dict = torch.load(os.path.join(f"{checkpoints_base_folder}/latest/netG_B_latest.pth"), map_location=self.device)
        netD_A_state_dict = torch.load(os.path.join(f"{checkpoints_base_folder}/latest/netD_A_latest.pth"), map_location=self.device)
        netD_B_state_dict = torch.load(os.path.join(f"{checkpoints_base_folder}/latest/netD_B_latest.pth"), map_location=self.device)

        self.netG_A.load_state_dict(netG_A_state_dict)
        self.netG_B.load_state_dict(netG_B_state_dict)
        self.netD_A.load_state_dict(netD_A_state_dict)
        self.netD_B.load_state_dict(netD_B_state_dict)

        # Update lr_scheduler back to where we were
        for _ in range(epoch):
            self.update_lr()

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

