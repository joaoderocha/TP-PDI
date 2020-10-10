import os
import re
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
import numpy as np
import math
from torchvision.transforms import ToTensor
from torch.autograd import Variable

ImageFile.LOAD_TRUNCATED_IMAGES = True
mtcnn = MTCNN(image_size=160, device='cuda')
model = InceptionResnetV1(pretrained='casia-webface').eval()  # Pre-trained on CASIA dataset

def printaMatris(matrix):
    for (i,vector) in enumerate(matrix):
        print(i)
        for (j,field) in enumerate(vector):
            print(j, field,end='')

def normalize_image(img):
    img = np.array(img)
    if img.shape[2] > 3:
        img = img[..., :3]
    return Image.fromarray(img, 'RGB')


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
        dist += (a - b) ** 2
    return math.sqrt(dist)


def getFiles(path, regex='*'):
    files = []
    for (dirpath, dirname, filenames) in os.walk(path):
        # print(dirpath)
        for file in filenames:
            # print(file)
            img = Image.open(dirpath + file)
            tensor = ToTensor()(img).unsqueeze(0)
            var = Variable(tensor)
            files.append({
                'nome': file.split('.')[0],
                'embedding': model(tensor),
                'img': img,
            })

    return files
print('fazendo vetor base')
vetorBase = getFiles('database/base_images/')
print('fazendo vetor teste')
vetorTeste = getFiles('database/test_images/')
threshold = 10
matrizConfusao = np.zeros((len(vetorBase), len(vetorBase)))
print('renomeando indiano')
cont = 0
for (i, fileI) in enumerate(vetorTeste): # vetor base => one shot
    min = sys.maxsize
    valorJ = -1
    menorJ = -1
    for (j, fileJ) in enumerate(vetorBase): # vetor teste => foto em grupo
        euclDist = euclidean_distance(fileI['embedding'], fileJ['embedding'])
        if euclDist < min:
            min = euclDist
            # menorJ = int((fileJ['nome'].split('_')))
            valorJ = int(fileJ['nome'])
    if min > threshold:
        continue
    # matrizConfusao[valorI][menorJ] += 1
    fileI['img'].save('cPessoas/teste2/{}_{}.jpg'.format(valorJ, cont))
    # os.remove(fileI['img'].filename)
    cont+=1


# print(matrizConfusao)
