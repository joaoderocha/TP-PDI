import time
from facenet_pytorch import InceptionResnetV1

from image_handler import getFiles
from threshold_calculus import calculate
from classifier import classify


def time_it_decorator(func, *args, **kwargs):
    start = time.time()
    function_return = func(*args, **kwargs)
    end = time.time()
    return function_return, end - start


if __name__ == '__main__':
    print('Calculando vetor de base...')
    (vetor_base, tempo) = time_it_decorator(getFiles, 'database/base_images/')
    print('levou: {} segundos'.format(tempo, 4))

    print('Calculando vetor de teste...')
    (vetor_teste, tempo) = time_it_decorator(getFiles, 'database/test_images/')
    print('levou: {} segundos'.format(tempo, 4))

    print('Recuperando modelo...')
    (model, tempo) = time_it_decorator(InceptionResnetV1(pretrained='casia-webface').eval)
    print('levou: {} segundos'.format(tempo, 4))

    print('Calculando threshold...')
    threshold = time_it_decorator(calculate, model, 'database/treshold/')  # 1.1139184950468057
    print('levou: {} segundos'.format(tempo, 4))

    print('Classificando alunos...')
    (indefinido, tempo) = time_it_decorator(classify(vetor_base, vetor_teste, threshold))
    print('levou: {} segundos'.format(tempo, 4))
