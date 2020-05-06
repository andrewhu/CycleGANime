import time
import torch
import datasets
import cycleganime 
import itertools
import utils

import cv2
if __name__ == "__main__":
    DATASET_PATH = "data/hatsune_miku"
    NUM_EPOCHS = 200
    BATCH_SIZE = 1
    GPU_IDS = [0]
    IM_SIZE = 256 # Input image size
    CROP_SIZE = 256 # Whatever image size fits in GPU memory

    # Which epoch/iter to resume training from
    RESUME_TRAINING = False
    START_EPOCH = 0
    START_ITER = 0

    # Dataloader for training
    dataset = datasets.UnpairedDataset(DATASET_PATH, im_size=IM_SIZE, crop_size=CROP_SIZE, phase='train', grayscale_A=True)
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Get test images for validation
    test_dataset = datasets.UnpairedDataset(DATASET_PATH, im_size=IM_SIZE, crop_size=IM_SIZE, phase='test', grayscale_A=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)
    test_data = list(itertools.islice(iter(test_dataloader), 4))

    # Define model
    model = cycleganime.CycleGANime(input_nc=3, output_nc=3, isTrain=True, gpu_ids=GPU_IDS)

    if RESUME_TRAINING:
        model.load_weights(epoch=START_EPOCH, it=START_ITER)
    else:
        START_EPOCH = -1

    # Keep track of our training loss
    losses = []
    plot_title = f"Image size {IM_SIZE}, Batch size {BATCH_SIZE}"

    for epoch in range(START_EPOCH+1,NUM_EPOCHS):
        epoch_start = time.time()
        for i, data in enumerate(dataloader):
            # Train model
            model.set_data(data)
            model.optimize_parameters()

            # Print losses
            if i % 10 == 0: 
                lG_A = model.loss_G_A.data
                lG_B = model.loss_G_B.data
                lD_A = model.loss_D_A.data
                lD_B = model.loss_D_B.data
                print(f"epoch {str(epoch).ljust(3, ' ')} - iter {str(i).ljust(4, ' ')} - G_A: {lG_A:.4f} - G_B: {lG_B:.4f} - D_A: {lD_A:.4f} - D_B: {lD_B:.4f}")

                # Append losses
                losses.append({
                    'G_A': lG_A,
                    'G_B': lG_B,
                    'D_A': lD_A,
                    'D_B': lD_B
                })

            if i % 10 == 0:
                # Run inference on test images
                print("Computing visuals...")
                model.compute_visuals(test_data, IM_SIZE, epoch, i)

                # Save model weights
                print("Saving model...")
                model.save(epoch, i)

        # Print epoch duration
        epoch_duration = time.time() - epoch_start
        print(f"Epoch completed in {epoch_duration:.0f}s")

        

        # Save loss plot to image
        print("Saving loss plot")
        utils.save_loss_plot(losses, plot_title)



        # Update learning rate at end of epoch
        print("Updating learning rate")
        model.update_lr()
        for G_pg, D_pg in zip(model.optimizer_G.param_groups, model.optimizer_D.param_groups):
            print(f"New learning rate: G: {G_pg['lr']}, D: {D_pg['lr']}")

