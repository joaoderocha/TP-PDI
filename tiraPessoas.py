import os
import re
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
import numpy as np
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True
mtcnn = MTCNN(image_size=160, margin=32, device='cuda', keep_all=True)
model = InceptionResnetV1(pretrained='casia-webface').eval()  # Pre-trained on CASIA dataset


def normalize_image(img):
    img = np.array(img)
    if img.shape[2] > 3:
        img = img[..., :3]
    return Image.fromarray(img, 'RGB')


def getPessoas(path, regex='*'):
    for (dirpath, dirname, filenames) in os.walk(path):
        for file in filenames:
            faces = mtcnn(normalize_image(Image.open(path+file)), save_path='cPessoas/rosto/{}'.format(file))


getPessoas('cPessoas/')