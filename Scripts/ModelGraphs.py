#!/usr/bin/python3

""" ---------------ModelGraphs.py----------------
| Purpose: To take the weights.npz and title2indx.json files and present
|            2 dimensional graphs for exploration and analysis.
| Input Arguments: 1) Path to the directory that the weights.npz and title2indx.json
|           files are in. These 2 files must be in the same directory and must
|           correspond to each other for the outcome to be intelligble.
|           2) -a gives the command to annotate the points on the graph. By
|               default this does not happen.
| History: Created on November 1, 2018 by Greg Colledge.
|       * November 7, 2018 - added the -a option to the command line so the
|           user can choose to show annotations. The default is that it does
|           not show labels. Also,  added an option to do a 3D graph with the
|           command line option '-3d'.
|
| Notes: The graphs may come with labels. You may have to use the magnifying
|           glass feature to read them.
"""

#------IMPORTS-----
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit as sigmoid
from datetime import datetime

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import sys
import string
import json

#-----LOAD MODEL FUNCTION-----
def load_model(directory):
    with open('%s/title2indx.json' % directory) as f:
        title2indx = json.load(f)
    npz = np.load('%s/weights.npz' % directory)
    W = npz['arr_0']
    V = npz['arr_1']
    return title2indx, W, V

#-----CHECK THE INPUT----
n_dim = 2;# default number of dimensions
if(len(sys.argv) < 2 and len(sys.argv) > 4):#2 to 4 arguments including script name
    print("Incorrect number of command line arguments.")
    exit();

if(len(sys.argv) == 2):
    annotate = False;
    n_dim = 2;
elif(len(sys.argv) == 3):
    if(sys.argv[2]=='-a'):
        annotate = True;
    elif(sys.argv[2]=='-3d'):
        n_dim = 3;
    else:
        print("Unrecognized command: %s" % sys.argv[2])
        print("Please see documentation and try again.")
        exit()
else:
    print("*****NOTE: Annotations can only be used in 2 dimensions.*****")
    annotate = True;
    n_dim = 2;


title2indx, W, V = load_model(sys.argv[1])
print("number of titles: %d" %len(title2indx))

#----- SET UP INPUTS FOR PLOTS-----
indx2Title = {}
for title, indx in enumerate(title2indx):
    indx2Title[indx] = title

titleList = list(range(len(title2indx)))
for i, t in enumerate(indx2Title):
    titleList[i] = t
#-----DO DIMENSION REDUCTION-----
#print statements allow you to know that the script is working
print("Computing PCA")
pca = PCA(n_components = n_dim)
W_pca = pca.fit_transform(W)
print("Computing T-SNE")
tsne = TSNE(n_components=n_dim)
W_tsne = tsne.fit_transform(W)

#-----SET UP PLOTS-----
if(n_dim==2):
    fig1 = plot.figure()
    #PCA plot
    # ax = fig.add_subplot(111, projection='3d')
    ax_pca = fig1.add_subplot(111)
    ax_pca.scatter(W_pca[:,0], W_pca[:,1])
    if(annotate==True):
        for i, label in enumerate(titleList):
            ax_pca.annotate(label, (W_pca[i,0], W_pca[i,1]))
    ax_pca.set_xlabel('X Label')
    ax_pca.set_ylabel('Y Label')
    ax_pca.set_title('PCA plot')

    #TSNE PLOT
    fig2 = plot.figure()
    ax_tsne = fig2.add_subplot(111)
    ax_tsne.scatter(W_tsne[:,0], W_tsne[:,1])
    if(annotate==True):
        for i, label in enumerate(titleList):
            ax_tsne.annotate(label, (W_tsne[i,0], W_tsne[i,1]))
    ax_tsne.set_xlabel('X Label')
    ax_tsne.set_ylabel('Y Label')
    ax_tsne.set_title('T-SNE plot')
    plot.show()
else:
    fig1 = plot.figure()
    #PCA plot
    ax_pca = fig1.add_subplot(111, projection='3d')
    ax_pca.scatter(W_pca[:,0], W_pca[:,1], W_pca[:,2])
    ax_pca.set_xlabel('X Label')
    ax_pca.set_ylabel('Y Label')
    ax_pca.set_zlabel('Z Label')
    ax_pca.set_title('PCA plot')

    #TSNE PLOT
    fig2 = plot.figure()
    ax_tsne = fig2.add_subplot(111, projection='3d')
    ax_tsne.scatter(W_tsne[:,0], W_tsne[:,1], W_tsne[:,2])
    ax_tsne.set_xlabel('X Label')
    ax_tsne.set_ylabel('Y Label')
    ax_pca.set_zlabel('Z Label')
    ax_tsne.set_title('T-SNE plot')
    plot.show()
