#!/usr/bin/python3

""" ---------------analogyTest.py----------------
| Purpose: To take the weights.npz and title2indx.json and find analogies in the
|           data set. Read the notes for more details.
| Input Arguments:
|   1) Path to the directory that the weights.npz and title2indx.json
|           files are in. These 2 files must be in the same directory and must
|           correspond to each other for the outcome to be intelligble.
|   2) title1 in the equation "title1 - title2 + title3 = ".
|           No parenthesis are needed.
|   3) title2 in the equation "title1 - title2 + title3 = ".
|           No parenthesis are needed.
|   4) title3 in the equation "title1 - title2 + title3 = ".
|           No parenthesis are needed.
| History: Created on November 1, 2018 by Greg Colledge.
|
| Notes: The graphs will come with the titles provided, the actual position of
|        the resultant vector and the titles of the closest vectors.
|       Also, The equation "title1 - title2 + title3 = " comes from the classic
|       word2vec example of "King - Man + Woman = " awhere the answer would be
|        "Queen". In this simple example the relationship is pulled out with the
|        subtraction and then reapplied to a new situation. This is what the
|       analogy function is trying to do/show.
|       An Example of the input line for this script is:
|       "python3 analogyTest.py /user/me/mydirectory web_developer software_engineer mechanical_engineer"

"""

#-----IMPORTS-----
import matplotlib.pyplot as plot
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
import sys
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


#-----LOAD MODEL FUNCTION-----
def load_model(directory):
    with open('%s/title2indx.json' % directory) as f:
        title2indx = json.load(f)
    npz = np.load('%s/weights.npz' % directory)
    W = npz['arr_0']
    V = npz['arr_1']
    return title2indx, W, V

#-----DEFINE ANALOGY FUNCTION -------
def analogy(positive1, negative1, negative2, title2indx, indx2title, W):
    V, D = W.shape

    #number of closest matches
    num_closest = 5

    print("--------------\ntesting: %s - %s + %s = " %(positive1, negative1, negative2))
    for t in (positive1, negative1, negative2):
        if t not in (title2indx):
            print(t + " is not in the used titles.")

    p1 = W[title2indx[positive1]]
    n1 = W[title2indx[negative1]]
    n2 = W[title2indx[negative2]]

    #the hope is that vec will be the same as gt
    vec = p1 - n1 + n2

    vectorList = list(range(num_closest))
    titleList = list(range(num_closest))
    titleList.append('Actual_Title')
    vectorList.append(vec)

    #using the cosine distance rather than the euclidian distance.
    distances = pairwise_distances(vec.reshape(1,D), W, metric='cosine').reshape(V)
    indx = distances.argsort()[:num_closest] #gets the index of the 5 closest vectors
    bestIndx = -1
    # remove the input titles as possibilities
    keep_out = [title2indx[t] for t in (positive1, negative1, negative2)]
    for r in indx:
        if r not in keep_out:
            bestIndx = r;
            break
#-----PRINT ANALOGY RESULTS-----
    indxCounter = 0
    for i in indx:
        print(indx2title[i], distances[i])
        vectorList[indxCounter] = W[i]
        titleList[indxCounter] = indx2title[i]
        indxCounter +=1

    for t in (positive1, negative1, negative2):
        vectorList.append(W[title2indx[t]])
        titleList.append(t)
#-----DIMENSION REDUCTION-----

    pca = PCA(n_components = 2)
    W_pca = pca.fit_transform(vectorList)
    pca3 = PCA(n_components = 3)
    W_pca3 = pca3.fit_transform(vectorList)

#-----PLOT ANALOGY RESULTS in 2D-----
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(W_pca[:,0], W_pca[:,1])
    for i, label in enumerate(titleList):
        ax.annotate(label, (W_pca[i,0], W_pca[i,1]))

    plot.show()

#-----PLOT ANALOGY RESULTS IN 3D-----
    fig3d = plot.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(W_pca3[:,0], W_pca[:,1], W_pca3[:,2])
    # for i, label in enumerate(titleList):
    #     ax.annotate(label, (W_pca3[i,0], W_pca3[i,1], W_pca3[i,2]))

    plot.show()

""" MAIN FUNCTION """
title2indx, W, V = load_model(sys.argv[1])
indx2title = {i:t for t, i in title2indx.items()}
analogy(sys.argv[2], sys.argv[3], sys.argv[4], title2indx, indx2title, W)
