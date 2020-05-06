import torch
import utils
import itertools
import numpy as np
import cv2
import random
import os
import glob

class CycleGANime():
    """Lineart colorization model based on CycleGAN"""
    def __init__(self, input_nc=3, output_nc=3, isTrain=False, gpu_ids=[], seed=420):
        torch.manual_seed(seed)

        self.isTrain = isTrain

        self.device = torch.device(f"cuda:{gpu_ids[0]}" if len(gpu_ids) > 0 else 'cpu')
        # self.device = torch.device('cuda:%d' % gpu_id if gpu_id is not None else 'cpu')
        # print("device:", self.device)

        # Hyperparameters
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lambda_idt = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.pool_size = 50     # Image buffer size
        self.G_n_blocks = 9     # Num of resnet blocks in generator

        # Define colorization generator network
        self.netG_A = utils.netG(input_nc, output_nc, 64, n_blocks=self.G_n_blocks, gpu_ids=gpu_ids)

        if self.isTrain:
            # Cycle consistency generator network
            self.netG_B = utils.netG(output_nc, input_nc, 64, n_blocks=self.G_n_blocks, gpu_ids=gpu_ids)

            # Define discriminator networks
            self.netD_A = utils.netD(output_nc, 64, n_layers=3, gpu_ids=gpu_ids)
            self.netD_B = utils.netD(input_nc, 64, n_layers=2, gpu_ids=gpu_ids)

            # Define image pools
            self.fake_A_pool = utils.ImagePool(self.pool_size)
            self.fake_B_pool = utils.ImagePool(self.pool_size)

            # Define loss functions
            self.criterionGAN = utils.GANLoss().to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # Define optimizers
            netG_params = itertools.chain(self.netG_A.parameters(), self.netG_B.parameters())
            netD_params = itertools.chain(self.netD_A.parameters(), self.netD_B.parameters())
            self.optimizer_G = torch.optim.Adam(netG_params, lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(netD_params, lr=0.0001, betas=(self.beta1, 0.999))

            # Learning rate schedulers
            self.scheduler_G = utils.get_lr_scheduler(self.optimizer_G)
            self.scheduler_D = utils.get_lr_scheduler(self.optimizer_D)

    def set_data(self, data):
        """Set input data"""
        self.real_A = data['A'].to(self.device)
        if self.isTrain:
            self.real_B = data['B'].to(self.device)

    def forward(self):
        """Forward pass"""
        self.fake_B = self.netG_A(self.real_A)
        if self.isTrain:
            self.inv_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.inv_B = self.netG_A(self.fake_A)

    def run_inference(self, data):
        with torch.no_grad():
            self.netG_A.eval()
            real_A = data['A'].unsqueeze(0)
            print(real_A.type())
            # print(real_A.size())
            fake_B = self.netG_A(real_A)
            return fake_B.permute(0,2,3,1).cpu().numpy()


    def backward_D(self, netD, real, fake):
        """Calculates discriminator loss"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()
        return self.loss_D

    def backward_D_A(self):
        """Calculate loss for discriminator A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate loss for discriminator B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D(self.netD_B, self.real_A, fake_A)

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
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def compute_visuals(self, test_data, sz, epoch, it):
        """Saves validation results"""
        with torch.no_grad():
            # Save image that the last iteration was trained on 
            # ref_im = self.real_B.permute(0,2,3,1).cpu().numpy()
            # assert train_data['A'].size()[0] == 8 and test_data['A'].size()[0] == 8
            im = np.zeros((sz*8,sz*8,3))

            # Compute test results
            for idx, data in enumerate(test_data):
                # print(type(data))
                self.set_data(data)
                self.forward()
                output_cpu = self.fake_B.permute(0,2,3,1).cpu().numpy()
                data_cpu = data['A'].permute(0,2,3,1).cpu().numpy()
                for i, img in enumerate(data_cpu):
                    im[idx*2*sz:idx*2*sz+sz, i*sz:i*sz+sz, :] = img
                for i, img in enumerate(output_cpu):
                    im[(idx*2+1)*sz:(idx*2+1)*sz+sz, i*sz:i*sz+sz, :] = img
            cv2.imwrite(f"results/result_{str(epoch).zfill(3)}_{str(it).zfill(4)}.jpg", cv2.cvtColor((im*255).astype(np.float32), cv2.COLOR_RGB2BGR))

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def save(self, epoch, it):
        """Save model weights"""
        folder_name = f"checkpoints/{str(epoch).zfill(3)}_{str(it).zfill(4)}" # save folder
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        G_A_filename = f"{folder_name}/netG_A_{str(epoch).zfill(3)}_{str(it).zfill(4)}.pth"
        G_B_filename = f"{folder_name}/netG_B_{str(epoch).zfill(3)}_{str(it).zfill(4)}.pth"
        D_A_filename = f"{folder_name}/netD_A_{str(epoch).zfill(3)}_{str(it).zfill(4)}.pth"
        D_B_filename = f"{folder_name}/netD_B_{str(epoch).zfill(3)}_{str(it).zfill(4)}.pth"
        torch.save(self.netG_A.state_dict(), G_A_filename)
        torch.save(self.netG_B.state_dict(), G_B_filename)
        torch.save(self.netD_A.state_dict(), D_A_filename)
        torch.save(self.netD_B.state_dict(), D_B_filename)
        # torch.save(self.netG_A.state_dict(), "checkpoints/netG_A_latest.pth")



    def load_weights_for_inference(self, weights_path):
        """Load generator A2B weights for inference"""
        print(weights_path)
        netG_A_state_dict = torch.load(weights_path, map_location=self.device)

        # We trained the network with DataParallel so we'll remove it here
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in netG_A_state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        self.netG_A.load_state_dict(new_state_dict)


    def load_weights(self, epoch, it):
        """Load all network weights to resume training"""
        
        epoch_iter = f"{str(epoch).zfill(3)}_{str(it).zfill(4)}"
        weights_path = f"checkpoints/{epoch_iter}"

        print(f"Loading weights from '{weights_path}'")

        netG_A_state_dict = torch.load(os.path.join(weights_path, f"netG_A_{epoch_iter}.pth"), map_location=self.device)
        self.netG_A.load_state_dict(netG_A_state_dict)
        if self.isTrain:
            netG_B_state_dict = torch.load(os.path.join(weights_path, f"netG_B_{epoch_iter}.pth"), map_location=self.device)
            self.netG_B.load_state_dict(netG_B_state_dict)

            netD_A_state_dict = torch.load(os.path.join(weights_path, f"netD_A_{epoch_iter}.pth"), map_location=self.device)
            self.netD_A.load_state_dict(netD_A_state_dict)

            netD_B_state_dict = torch.load(os.path.join(weights_path, f"netD_B_{epoch_iter}.pth"), map_location=self.device)
            self.netD_B.load_state_dict(netD_B_state_dict)

        # Update lr_scheduler back to where we were
        for _ in range(epoch):
            self.update_lr()



        
