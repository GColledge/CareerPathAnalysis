#!/usr/bin/python3

"""--------------titleClustering.py---------------------
| Purpose:  To help vizualize the possible clustering of the career paths data.
| Input Argument:   1) Path to the directory that the weights.npz and title2indx.json
|           files are in. These 2 files must be in the same directory and must
|           correspond to each other for the outcome to be intelligble.
|           2) The number of desired clusters to show. This is only used if the
|               graph of clusters and cluster centers is to be shown.
|           3) The number of dimensions for the clusters to be plotted in. This
|               is only used if the second argument is given. See (2) above.
|           4) a '-d' given as the last argument indicates that the dendogram
|               should be displayed. This comes as the last argument regardless
|               of wether arguments (2) and (3) are given.
| History:  Created November 12, 2018 by Greg Colledge
|           * November 21, 2018 - Added the input error checking as well as
|               giving the options to just show the dendogram.
"""
from MyFunctions import load_model
from sklearn.cluster import KMeans #this is currently an arbitrary choice.
#More thought should be put into what model to use.
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


#-----INPUT CHECKING-----
dend = False;#default to not displaying the dendogram
cluster = True; #defualt to display cluster graph
if(len(sys.argv) < 3 and len(sys.argv) > 5):#3 to 5 arguments including script name
    print("Incorrect number of command line arguments.")
    exit();
else:
    #----LOAD MODEL-----
    try:
        directory = sys.argv[1]
        title2indx, W, V = load_model(directory)
    except FileNotFoundError as e:
        print("The file ")
        print(sys.argv[1])
        print("was not found.")
        exit();
if(sys.argv[2]=='-d'):#for 3 arguments (just dendogram)
    dend=True
    cluster=False
elif(len(sys.argv)==4):#4 arguments (just cluster graph)
    if(sys.argv[3] != '2' and sys.argv[3] != '3'):
        print(sys.argv[3])
        print("Incorrect number of command line arguments for clustering graph. \n"
        +"Needs a number of clusters and a number of dimensions to plot in (2 or 3).")
        exit();
elif(len(sys.argv)==5):#5 arguments (for when both dendogram and cluster graph is shown.)
    if(sys.argv[3] != '2' and sys.argv[3] != '3'):
        print(sys.argv[3])
        print("Incorrect number of command line arguments for clustering graph. \n"
        +"Needs a number of clusters and a number of dimensions to plot in (2 or 3).")
        exit();
    elif(sys.argv[4]=='-d'):
        dend==True
    else:
        print("Use -d as the final command line argument if a dendogram is desired.")
        dend==False
else:
    print("Input Error occured")
    print(sys.argv)
    exit();



indx2Title = {}
for indx, title in enumerate(title2indx):
    indx2Title[int(indx)] = title
if(dend == False):
    num_of_clusters = int(sys.argv[2])
    n_dim = int(sys.argv[3]); #the number of dimensions to reduce the data down to.

    kmeans = KMeans(n_clusters=num_of_clusters, random_state=4)
    kmeans.fit(W)

    #-----DO DIMENSION REDUCTION-----
    centers = kmeans.cluster_centers_
    #print statements allow you to know that the script is working
    print("Computing PCA")
    pca = PCA(n_components = n_dim)
    W_pca = pca.fit_transform(W)
    pca_centers = pca.transform(centers)

    if(n_dim==2):
        fig1 = plot.figure()
        #PCA plot
        ax_pca = fig1.add_subplot(111)
        ax_pca.scatter(W_pca[:,0], W_pca[:,1], color='blue')
        ax_pca.scatter(pca_centers[:,0], pca_centers[:,1], color='red')
        ax_pca.set_title('PCA compressed data with KMeans centers')

        plot.show()
    elif(n_dim==3):
        fig1 = plot.figure()
        #PCA plot
        ax_pca = fig1.add_subplot(111, projection='3d')
        ax_pca.scatter(W_pca[:,0], W_pca[:,1], W_pca[:,2], color='blue')
        ax_pca.scatter(pca_centers[:,0], pca_centers[:,1], pca_centers[:,2], color='red')
        ax_pca.set_title('PCA compressed data with KMeans centers')

        plot.show();
else:
    #-----DENDOGRAM-----#
    def leaf_labeler(id):
        if id < len(indx2Title):
            return indx2Title[id];
        else:
            return id

    plot.figure()
    plot.title("Job Title Dendograms")
    # dend = shc.dendrogram(shc.linkage(W, method='complete'), p=5, truncate_mode='level', leaf_rotation=90)
    print("Creating Dendogram")
    dend = shc.dendrogram(shc.linkage(W, method='weighted'), leaf_rotation=45, leaf_label_func=leaf_labeler)

    plot.show()
    #-----Hierarchal clustering-----
    # clustering = AgglomerativeClustering(n_clusters=num_of_clusters, affinity='cosine', linkage='complete')
    # clustering.fit_predict(W)
    # print(cluster)
