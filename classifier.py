import sys
from threshold_calculus import euclidean_distance


def classify(base_vector, test_vector,  threshold, save_path='cPessoas/test2/'):
    cont = 0
    for (i, fileI) in enumerate(test_vector):  # vetor base => one shot
        min = sys.maxsize
        valorJ = -1
        for (j, fileJ) in enumerate(base_vector):  # vetor teste => foto em grupo
            euclDist = euclidean_distance(fileI['embedding'], fileJ['embedding'])
            if euclDist < min:
                min = euclDist
                valorJ = int(fileJ['nome'])
        if min > threshold:
            fileI['img'].save(save_path+'NINGUEM_{}.jpg'.format(cont))
            cont += 1
            continue

        fileI['img'].save(save_path+'{}_{}.jpg'.format(valorJ, cont))
        cont += 1