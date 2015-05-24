#!/bin/env python2


from os import listdir
from os.path import isfile, join
import cv2
import sift
import time
import random
import glob
import pickle
import numpy as np


def getsamples(descriptors):
    samples = random.sample(descriptors, 100000)
    f = open('data/samples', 'w')
    pickle.dump(samples, f)
    f.close()
    return samples

images = [f for f in listdir('../datasets/oxbuild_images') if isfile(join('../datasets/oxbuild_images', f))]
queries = glob.glob('../datasets/gt_files_170407/*query.txt')
queryimages = []

for queryfile in queries:
    f = open(queryfile, 'r')
    queryimage = (f.readline().split(' ')[0] + '.jpg')[5::]
    queryimages.append(queryimage)
    f.close()

alldescriptors = None
count = 0
samples = None

if isfile('data/descriptors'):
    print('Found samples file, preloading...')
    f = open('data/descriptors', 'r')
    alldescriptors = pickle.load(f)
    f.close()
    if isfile('data/samples'):
        f = open('data/samples', 'r')
        samples = pickle.load(f)
        f.close()
    else:
        samples = getsamples(alldescriptors)
else:
    alldescriptors = []
    for image in images:
        if image not in queryimages:
            start = time.time()
            filepath = join('../datasets/oxbuild_images', image)
            img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            shape = img.shape
            maxshape = max(img.shape)
            if maxshape > 640:
                factor = 640.0 / maxshape
                newshape = tuple([int(factor * i) for i in img.shape])
                newshape = newshape[::-1]
                img = cv2.resize(img, newshape)
            else:
                newshape = img.shape

            keypoints, descriptors = sift.getdescriptors(img)
            if descriptors is not None:
                for d in descriptors:
                    alldescriptors.append(d)
            count += 1
            end = time.time()
            print(str(count) + "/" + str(len(images)) + " - " + str(end - start))
            # keyimg = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow('keypoints', keyimg)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     break
        else:
            print('Image ' + image + ' is a query image')
    f = open('data/descriptors', 'w')
    pickle.dump(alldescriptors, f)
    f.close()
    print('Total number of descriptors: ' + str(len(alldescriptors)))
    samples = getsamples(alldescriptors)

clusters = [pow(2, i) for i in [6, 7, 8]]
data = np.array(samples)

kmeansresults = []
if isfile('data/64'):
    for k in clusters:
        f = open('data/' + str(k), 'r')
        kmeansresults.append(pickle.load(f))
        f.close()
else:
    for k in clusters:
        print(k)
        start = time.time()
        clusters = cv2.kmeans(data, K=k,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001),
                              attempts=10,
                              flags=cv2.KMEANS_RANDOM_CENTERS)
        end = time.time()
        print end - start
        f = open('data/' + str(k), 'w')
        pickle.dump(clusters, f)
        f.close()
        kmeansresults.append(clusters)

print('Done')
print(kmeansresults)

