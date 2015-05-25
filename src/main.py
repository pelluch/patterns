#!/bin/env python2

from os import listdir
from os.path import isfile, join
import cv2
import utils
import time
import numpy as np
import pickle

images = [f for f in listdir('../datasets/oxbuild_images') if isfile(join('../datasets/oxbuild_images', f))]
queryimages = utils.getqueryimages()
alldescriptors = None
count = 0
samples = None

startall = time.time()
alldescriptors = []

if isfile('data/samples'):
    samples = utils.getsamples(None)
else:
    for image in images:
        if image not in queryimages:
            start = time.time()
            filepath = join('../datasets/oxbuild_images', image)
            img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img = utils.resize(img)
            keypoints, descriptors = utils.getdescriptors(img)
            if descriptors is not None:
                for d in descriptors:
                    alldescriptors.append(d)
            count += 1
            end = time.time()
            print(str(count) + "/" + str(len(images)) + " - " + str(end - start))
        else:
            print('Image ' + image + ' is a query image')
    samples = utils.getsamples(alldescriptors)

endall = time.time()
print('Total time: ' + str(endall - startall))
# print('Total number of descriptors: ' + str(len(alldescriptors)))
# samples = utils.getsamples(alldescriptors)
data = np.array(samples)

# Array of 3 elements for each cluster
# Each element has the format (value, bestlabels, centers)

kmeansresults = utils.getclusters(data)

print('Done')
# print(kmeansresults)

patches = utils.getquerypatches()
patchdescriptors = []
for patch in patches:
    keypoints, descriptors = utils.getdescriptors(patch[1])
    patchdescriptors.append(descriptors)

vlad_descriptors = []
clusters = [pow(2, i) for i in [6, 7, 8]]

for i, result in enumerate(kmeansresults):
    if len(alldescriptors) == 0:
        counter = 0
        for image in images:
            if image not in queryimages:
                filepath = join('../datasets/oxbuild_images', image)
                img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                img = utils.resize(img)
                keypoints, descriptors = utils.getdescriptors(img)
                vlad = np.zeros((clusters[i], 128))
                if descriptors is not None:
                    for descriptor in descriptors:
                        idx, closest = utils.find_nearest(result[2], descriptor)
                        diff = descriptor - closest
                        vlad[idx] = vlad[idx] + diff
                vlad = vlad.flatten()
                vlad /= np.linalg.norm(vlad)
                vlad_descriptors.append(vlad)
                print(len(vlad))
            counter += 1
            print(counter)
    f = open('data/vlad_' + str(clusters[i]), 'w')
    pickle.dump(vlad_descriptors, f)
    f.close()
