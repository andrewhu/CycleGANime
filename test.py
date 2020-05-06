import time
import torch
import datasets
import cycleganime 
import itertools


if __name__ == "__main__":
    DATASET_PATH = "data/hatsune_miku"
    MODEL_PATH = "checkpoints/net_G_A_latest.pth" # Saved Generator model path
    NUM_EPOCHS = 200
    BATCH_SIZE = 4
    GPU_ID = 0

    # Dataloader for testing
    dataset = datasets.UnpairedDataset(DATASET_PATH, phase='test', grayscale_A=True)
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Define model
    model = cycleganime.CycleGANime(input_nc=3, output_nc=3, isTrain=False, gpu_id=GPU_ID)
    model.load_weights(DATASET_PATH)

    model.compute_visuals(test_data, actual_iter)



