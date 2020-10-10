import os
import math
from PIL import Image, ImageFile
from torchvision.transforms import ToTensor


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
        dist += (a - b) ** 2
    return math.sqrt(dist)


def calculate(model, path):
    maximum = -1
    for (dirpath, dirname, filenames) in os.walk(path):
        if not filenames:
            continue
        count1 = 0
        count2 = 0
        for img_base in filenames:
            count1 += 1
            for img_target in filenames:
                if img_target == img_base:
                    continue
                if count1 > count2:
                    base = Image.open(dirpath + '/' + img_base)
                    target = Image.open(dirpath + '/' + img_target)
                    base_emb = model(ToTensor()(base).unsqueeze(0))
                    target_emb = model(ToTensor()(target).unsqueeze(0))
                    dist = euclidean_distance(base_emb, target_emb)
                    maximum = max(maximum, dist)
                    count2 += 1
    return maximum