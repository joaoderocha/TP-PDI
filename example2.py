# SAMPLE CODE 2
# Similarity threshold calculation

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
import numpy as np
import os
import math
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Calcular threshold
threshold = None
PATH = '<Insert training dataset directory>'  # Terá que conter UMA imagem de cada aluno (imagens base do one-shot learning)

def value(tensor_elem):
    return tensor_elem.detach().numpy()

def tensor_to_array(tensor):
    arr = []
    for t in tensor[0]:
        arr.append(value(t))
    return arr

def euclidean_distance(tensor_1, tensor_2):
    t1 = tensor_to_array(tensor_1)
    t2 = tensor_to_array(tensor_2)
    dist = 0.0
    for a, b in zip(t1, t2):
        dist += (a - b)**2        
    return math.sqrt(dist)

def normalize_image(img):
    img = np.array(img)
    if img.shape[2] > 3:
        img = img[..., :3]    
    return Image.fromarray(img, 'RGB')

# Faces com distância > threshold serão consideradas diferentes.
def similarity_threshold(mtcnn, model, PATH):
    minimum = sys.maxsize
    for img_base in os.listdir(PATH):
        for img_target in os.listdir(PATH):
            if img_target == img_base: 
                continue
            base       = mtcnn(normalize_image(Image.open(PATH + '/' + img_base)))
            target     = mtcnn(normalize_image(Image.open(PATH + '/' + img_target)))
            base_emb   = model(base.unsqueeze(0))
            target_emb = model(target.unsqueeze(0))
            dist       = euclidean_distance(base_emb, target_emb)
            minimum    = min(minimum, dist)
    return minimum
            

#cur = os.getcwd()  # Get current working directory

mtcnn = MTCNN(image_size=160, margin=32, device='cuda')
model = InceptionResnetV1(pretrained='casia-webface').eval()  # Pre-trained on CASIA dataset

threshold = similarity_threshold(mtcnn, model, PATH)

print('Threshold: ' + str(threshold))
