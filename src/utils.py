
import cv2
import pickle
import random
from os.path import isfile, join
import glob
import numpy as np
import sys

def getdescriptors(image):
    sift = cv2.SIFT()
    return sift.detectAndCompute(image, None)

def getsamples(descriptors):
    if isfile('data/samples'):
        f = open('data/samples', 'r')
        samples = pickle.load(f)
        f.close()
        return samples
    else:
        samples = random.sample(descriptors, 100000)
        f = open('data/samples', 'w')
        pickle.dump(samples, f)
        f.close()
        return samples


def getclusters(descriptors):
    kmeansresults = []
    clusters = [pow(2, i) for i in [6, 7, 8]]
    if isfile('data/64'):
        for k in clusters:
            f = open('data/' + str(k), 'r')
            kmeansresults.append(pickle.load(f))
            f.close()
    else:
        for k in clusters:
            print(k)
            clusters = cv2.kmeans(descriptors, K=k,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001),
                                  attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)
            f = open('data/' + str(k), 'w')
            pickle.dump(clusters, f)
            f.close()
            kmeansresults.append(clusters)
    return kmeansresults

def getqueryimages():
    queryimages = []
    queries = glob.glob('../datasets/gt_files_170407/*query.txt')
    for queryfile in queries:
        f = open(queryfile, 'r')
        queryimage = (f.readline().split(' ')[0] + '.jpg')[5::]
        queryimages.append(queryimage)
        f.close()
    return queryimages

def resize(img):
    maxshape = max(img.shape)
    if maxshape > 640:
        factor = 640.0 / maxshape
        newshape = tuple([int(factor * i) for i in img.shape])
        newshape = newshape[::-1]
        resized = cv2.resize(img, newshape)
        return resized
    else:
        return img

def getquerypatches():
    patches = []
    queries = glob.glob('../datasets/gt_files_170407/*query.txt')
    for queryfile in queries:
        f = open(queryfile, 'r')
        data = f.readline().split(' ')
        f.close()
        queryimage = (data[0] + '.jpg')[5::]
        x1 = int(float(data[1]))
        y1 = int(float(data[2]))
        x2 = int(float(data[3]))
        y2 = int(float(data[4]))
        img = cv2.imread(join('../datasets/oxbuild_images', queryimage), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        img = img[y1:y2, x1:x2]
        patches.append((queryimage, img))
    return patches

def find_nearest(dictionary, vector):
    dictionary = np.array(dictionary)
    best_index = -1
    vector = np.array(vector)
    best_distance = sys.float_info.max
    closest = None
    for i, word in enumerate(dictionary):
        v = np.array(word)
        diff = vector - word
        dist = np.linalg.norm(diff)
        if dist < best_distance:
            closest = word
            best_distance = dist
            best_index = i
    return tuple([best_index, closest])
