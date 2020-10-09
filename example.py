# SAMPLE CODE
# Euclidean distances using two people, one of them as base model.

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
import numpy as np
import os
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# TODO: Calcular threshold
threshold = None

cur = os.getcwd()  # Get current working directory

mtcnn = MTCNN(image_size=160, margin=32, device='cuda')
model = InceptionResnetV1(pretrained='casia-webface').eval()  # Pre-trained on CASIA dataset

img_base = normalize_image(Image.open(cur + '/dataset/test_images/sample.jpg'))  # Image base for comparison (Kate Siegel)
img1     = normalize_image(Image.open(cur + '/dataset/test_images/siegel.jpg'))  # Kate Siegel
img2     = normalize_image(Image.open(cur + '/dataset/test_images/jolie.jpg'))   # Angelina Jolie

# Get cropped and prewhitened image tensor
face_base = mtcnn(img_base, save_path=cur + '/dataset/saved_files/base.jpg')
face1     = mtcnn(img1    , save_path=cur + '/dataset/saved_files/siegel_c.jpg')
face2     = mtcnn(img2    , save_path=cur + '/dataset/saved_files/jolie_c.jpg')

# Calculate embedding (unsqueeze to add batch dimension)
img_base_embedding = model(face_base.unsqueeze(0))
img_embedding1     = model(face1.unsqueeze(0))
img_embedding2     = model(face2.unsqueeze(0))

print('===== Euclidean distances =====')
print('Kate Siegel vs Kate Siegel: ', end='')
print(euclidean_distance(img_base_embedding, img_embedding1))
print('Kate Siegel vs Angelina Jolie: ', end='')
print(euclidean_distance(img_base_embedding, img_embedding2))


# =============================================================================
# OUTPUT:
# ===== Euclidean distances =====
# Kate Siegel vs Kate Siegel: 0.7633472480452842
# Kate Siegel vs Angelina Jolie: 0.9349967981526041
# =============================================================================
