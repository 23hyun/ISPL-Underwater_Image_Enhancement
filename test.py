import torch
import torchvision
import glob
import os
import model
import numpy as np
from PIL import Image

def Img_Loader(image_path):
    img_haze = Image.open(image_path)
    img_haze = (np.asarray(img_haze) / 255.0)
    img_haze = torch.from_numpy(img_haze).float()
    img_haze = img_haze.permute(2, 0, 1)


    img_haze = img_haze.cuda().unsqueeze(0)
    return img_haze

def dehaze_image(image_path):
    Transmission_Net = model.Transmission_Net().cuda()
    Transmission_Net.load_state_dict(torch.load('checkpoint path here'))

    # 모드 변경
    Transmission_Net.eval()

    test_path = 'test result path here'

    os.makedirs(test_path, exist_ok=True)


    img_haze = Img_Loader(image_path)


    with torch.no_grad():
        air_map = Transmission_Net(img_haze)
  
    torchvision.utils.save_image(air_map, test_path + "\\" + image_path.split("\\")[-1])

    del Transmission_Net
    torch.cuda.empty_cache()

if __name__ == '__main__':
    test_list = glob.glob("test image path here/*")

    for image in test_list:
        dehaze_image(image)
        print(image, "done!")

