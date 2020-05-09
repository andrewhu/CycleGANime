import time
import torch
import cycleganime 
import itertools
import utils
import random

import cv2
if __name__ == "__main__":
    DATASET_PATH = "data/pink_hair"
    NUM_EPOCHS = 200
    BATCH_SIZE = 1
    GPU_ID = 0
    IM_SIZE = 512 # Input image size
    CROP_SIZE = 512 # If you're running out of gpu memory, set this to something smaller than IM_SIZE

    SEED = 12 # me lucky number :))
    utils.set_seed(SEED)

    RESUME_TRAINING = False
    START_EPOCH = 0 # Lets learning rate scheduler know which epoch we're starting from

    # Dataloader for training
    train_dataloader = utils.get_dataloader(DATASET_PATH, phase='train', batch_size=BATCH_SIZE, im_size=IM_SIZE, crop_size=CROP_SIZE, num_workers=2)

    # Get test images for validation
    test_dataloader = utils.get_dataloader(DATASET_PATH, phase='test', batch_size=8, im_size=IM_SIZE, crop_size=CROP_SIZE, num_workers=1)
    test_data = list(itertools.islice(iter(test_dataloader), 4))

    # Define model
    model = cycleganime.CycleGANime(input_nc=3, output_nc=3, gpu_id=GPU_ID)

    # Load previous weights (or not)
    if RESUME_TRAINING:
        model.load_weights(epoch=START_EPOCH)
    else:
        START_EPOCH = 0

    # Keep track of our training loss
    loss_history = []
    plot_title = f"Image size {IM_SIZE}, Batch size {BATCH_SIZE}"

    for epoch in range(START_EPOCH,NUM_EPOCHS):
        epoch_start = time.time()
        for iteration, data in enumerate(train_dataloader):
            period_start = time.time()

            # Train model
            model.set_data(data)
            model.optimize_parameters()

            # Print losses
            if iteration % 10 == 0: 
                period_duration = time.time()-period_start
                period_start = time.time()
                losses = model.get_losses()
                print("epoch %s - iter %s - G_A %.4f - G_B %.4f - D_A %.4f - D_B %.4f - %.2f it/s" % 
                (str(epoch).ljust(3, ' '), str(iteration).ljust(4, ' '), losses['G_A'], losses['G_B'], losses['D_A'], losses['D_B'], period_duration/10))

                # Append losses
                loss_history.append(losses)

            if iteration % 100 == 0:
                # Run inference on test images
                print("Computing visuals...", end=' ')
                model.compute_visuals(test_data, IM_SIZE, epoch, iteration)
                print("done")

                # Save model weights
                print("Saving model...", end=' ')
                model.save_weights(epoch, iteration)
                print("done")

        # Update learning rate at end of epoch
        print("Updating learning rate...", end=' ')
        model.update_lr()
        print("done")
        for G_pg, D_pg in zip(model.optimizer_G.param_groups, model.optimizer_D.param_groups):
            print(f"New learning rate: G: {G_pg['lr']}, D: {D_pg['lr']}")

        # Print epoch duration
        epoch_duration = time.time() - epoch_start
        print(f"Epoch completed in {epoch_duration:.0f}s")

        # Save loss plot to image
        print("Saving loss plot...", end=' ')
        utils.save_loss_plot(loss_history, plot_title)
        print("done")
