import torch
from torch.optim import lr_scheduler
import random
import numpy as np
import matplotlib.pyplot as plt
import unpaired_dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataloader(dataset_path, phase, batch_size, im_size, crop_size, num_workers):
    dataset = unpaired_dataset.UnpairedDataset(dataset_path, im_size=im_size, crop_size=crop_size, phase=phase)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader


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


############# Loss plot saver #################################
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
