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
#Input: Directory -  (string) the directory that the json and weights file reside. They
#           must be together in the same directory. These are the output files of the
#           word2vec jupyter notebooks written by Greg Colledge.
#Returns: title2indx - a dictionary with keys that are titles (strings) and
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

#-----MAKE TEST FILE Function-----
# Input:    inputFile - (string) the full file path to the file to be used as
#               input. This must be a text file in the correct format.
#           outputFile - (string) the file name of the output file including the
#                file extension. This does not need to include the file path.
# Output:   a file with each line in the following format:
#               final_job_title,previous_job_title  another_job_title   etc.
#           each line will represent one persons career path. If not specified,
#           the file will be saved in the same folder as the file being run and
#           will take the name given by the outputFile argument.
# Returns:  True if no errors occured. False Otherwise.
def makeTestFile(inputFile, outputFile):
    with open(inputFile, 'r') as datafile:
        newFile = open(outputFile, 'w+')
        for line in datafile.readlines():
            line = line.split();
            if len(line) > 2:
                newFile.write(line[-1])
                newFile.write(',')
                for t in range(0, len(line) - 1 , 1):
                    newFile.write(line[t])
                    newFile.write('\t')
                newFile.write('\n')

        newFile.close()
    return True

#-----CONVERT STRINGS TO INDICES Function-----
#This is a specialized function specific to the models and files used in the
#Career Analysis. It takes training or testing files (in string format) and
#creates the same file with indices into the given model (in place of strings).
# Input:    inputFile - the training or test file in string format. This should
#               be the full path to the file.
#           outputFile - the name of the file to output as a string.
#           modelDirectory - the path (as a string) to the model to be used.
## Output:   a file with each line in the following format:
#               previous_job_index  another_job_index   etc.,final_job index
#           each line will represent one persons career path. If not specified,
#           the file will be saved in the same folder as the file being run and
#           will take the name given by the outputFile argument.
# Returns:  True if no errors occured. False Otherwise.
def convertStrings2Indices(inputFile, outputFile, modelDirectory):
    title2indx, W, V = load_model(modelDirectory);
    with open(inputFile, 'r') as datafile:
        newFile = open(outputFile, 'w+');
        for line in datafile.readlines():
            line = line.split(',')
            assert(len(line)==2);
            ans = line[0]
            path = line[1]
            path = path.split()
            for each in path:
                try:
                    newFile.write(str(title2indx[each]))
                except KeyError:
                    newFile.write(str(len(title2indx) - 1))
                newFile.write('\t')
            newFile.write(',')
            try:
                newFile.write(str(title2indx[each]))
            except KeyError:
                newFile.write(str(len(title2indx) - 1))
            newFile.write('\n')
        newFile.close();
    return True
