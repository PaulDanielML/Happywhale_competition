import os
from pathlib import Path
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import glob

from U2Net.data_loader import RescaleT
from U2Net.data_loader import ToTensor
from U2Net.data_loader import ToTensorLab
from U2Net.data_loader import SalObjDataset

from U2Net.model import U2NET

from tqdm import tqdm
from src.utils.common import INPUT_DATA_DIR, OUTPUT_DATA_DIR, BASE_DIR

INPUT_IMAGES = "train"


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


INPUT_DIR = INPUT_DATA_DIR / f"{INPUT_IMAGES}_images"
OUTPUT_DIR = OUTPUT_DATA_DIR / "cropped" / INPUT_IMAGES
MODEL_PATH = BASE_DIR / "U2Net" / "saved_models" / "u2net" / "u2net.pth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

img_name_list = glob.glob(str(INPUT_DIR) + os.sep + "*")[0]

test_salobj_dataset = SalObjDataset(
    img_name_list=img_name_list,
    lbl_name_list=[],
    transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
)
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(MODEL_PATH))
    net.cuda()
else:
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
net.eval()

bar = tqdm(enumerate(test_salobj_dataloader))

for i_test, data_test in bar:
    print(data_test["image"])
