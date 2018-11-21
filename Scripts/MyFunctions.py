"""----------------------MY FUNCTIONS-----------------------
This is a file of helper functions that can be imported into notebopoks or other
 scripts. These functions are designed to be used with the LinkedIn data that
 was scraped by Nick Sullivan and in the format that he decided on.
"""

from json import load as jload
import numpy as np

#A function specific to this data set. It takes the career path, makes a list
#of job titles while removing the underscores.
def prettify(careerPath):
    careerPathList = careerPath.split();
    for title in careerPathList:
        careerPathList[careerPathList.index(title)] = title.replace('_', ' ');
    return careerPathList;

#-----LOAD MODEL FUNCTION-----
#Input: Directory -  the directory that the json and weights file reside. They
#           must be together in the same directory. These are the output files of the
#           word2vec jupyter notebooks written by Greg Colledge.
#Output: title2indx - a dictionary with keys that are titles (strings) and
#           values that are the indexes into the title embeddings matrix.
#       W - The weights of the first hidden layer of the model. The matrix is a
#            numpy.ndarray with shape NxD where N is the number of unique job
#           titles used, and D isthe dimensionality of the space.
#       V - the weights of the second hidden layer. W does not have to be the
#           definnitive definition for the word embedding. V is the weights of
#            another layer given as another numpy.ndarray of size DxN. This can
#           be used to create a better title embedding. a common embedding is
#           given as (W + V.T)/2 which is essentially the average of the two
#           hidden layers in the nueral network.
def load_model(directory):
    with open('%s/title2indx.json' % directory) as f:
        title2indx = jload(f)
    npz = np.load('%s/weights.npz' % directory)
    W = npz['arr_0']
    V = npz['arr_1']
    return title2indx, W, V
