import torch
import cycleganime
import cv2
import time

INPUT_IM_PATH = "splash.jpg"
WEIGHTS_PATH = "checkpoints/000_0000/netG_A_000_0000.pth"


if __name__ == "__main__":

    # Init model in cpu mode
    print("Initializing model")
    model = cycleganime.CycleGANime() 
    
    # Load weights
    print("Loading weights")
    model.load_weights_for_inference(WEIGHTS_PATH)

    # model.eval()

    # Read in image, resize, then convert to tensor
    print("Reading in image")
    im = cv2.imread(INPUT_IM_PATH)
    im = cv2.resize(im, (256,256))
    im = torch.Tensor(im).permute(2,0,1)
    input_data = {'A': im}
    
    # Run inference
    start = time.time()
    result =  model.run_inference(input_data)
    print(time.time()-start)
    print(result.shape)

    cv2.imwrite("Test.jpg", result[0]*255)





