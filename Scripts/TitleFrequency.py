#!/usr/bin/python3

"""
| Purpose:  To Give a frequency count of the titles and give information to help decide on a cutoff.
|
| History:  Created on November 21, 2018 by Greg Colledge
|
"""

import sys
import operator
import statistics as stats

#-----READ IN DATA-----
titleFreq = {}
with open(sys.argv[1],'r') as dataFile:
    for line in dataFile:
        for title in line.split():
            if(title not in titleFreq):
                titleFreq[title] = 0;
            titleFreq[title] += 1;
print("Finished counting titles")

#-----SortedList----
sortedList = sorted(titleFreq.items(), key=operator.itemgetter(1))
titles = []
freq = []
for (title, num) in sortedList:
    titles.append(title)
    freq.append(num)

keepList = [] #this is the list of (title, freq) tuples that pass the criterion.
if(sys.argv[2]=='-f'):
    #frequency cut off
    target = int(sys.argv[3])
    print("length of sorted: %s" %str(len(sortedList)))
    print(str(sortedList[0]))
    print(str(sortedList[-1]))
    for i in range(len(sortedList)-1,0,-1):
        print("value to compare: %s" %str(sortedList[i]))
        if sortedList[i][1] > target:
            print(sortedList[i])
            keepList.append(sortedList[i]);
        else:
            print(target)
            break;

elif(sys.argv[2]=='-c'):
    #title count cutoff
    target = int(sys.argv[3])
    for i in range(len(sortedList)):
        keepList.append(sortedList[i]);


print("length of keepList: %d" %len(keepList))
keepFreq = []
keepTitles = []
for each in keepList:
    keepFreq.append(each[1])
    keepTitles.append(each[0])
#-----Stats stuff-----
avgFreq = sum(freq) / len(titleFreq);
stdDev_of_Freq = stats.stdev(freq)
avgKeepFreq = stats.mean(keepFreq)
sd_of_Keep = stats.stdev(keepFreq)

#-----OUTPUT-----
for i in range(20):
    print(str(freq[i]) + "\t" + str(titles[i]))

print(len(titleFreq))
print("average frequency: %d" % avgFreq)
print("std deviation: %d" % stdDev_of_Freq)
print("average Kept frequency: %d" % avgKeepFreq)
print("std deviation of Kept: %d" % sd_of_Keep)
for i in range(1,11):
    print(sortedList[-i])
