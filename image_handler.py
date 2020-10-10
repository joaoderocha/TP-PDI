import os
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageFile
import numpy as np
from torchvision.transforms import ToTensor


ImageFile.LOAD_TRUNCATED_IMAGES = True
model = InceptionResnetV1(pretrained='casia-webface').eval()  # Pre-trained on CASIA dataset


def normalize_image(img):
    img = np.array(img)
    if img.shape[2] > 3:
        img = img[..., :3]
    return Image.fromarray(img, 'RGB')


def getFiles(path):
    files = []
    for (dirpath, dirname, filenames) in os.walk(path):
        for file in filenames:
            img = Image.open(dirpath + file)
            tensor = ToTensor()(img).unsqueeze(0)
            files.append({
                'nome': file.split('.')[0],
                'embedding': model(tensor),
                'img': img,
            })

    return files