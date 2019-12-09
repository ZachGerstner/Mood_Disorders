import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from random import randint
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.externals import joblib


clf = joblib.load('Mood_disorder_45')
clf.predict()
'''
data = pd.read_csv('GTEx_v7_brain_subGEM-log-no.txt',sep='\t')
data = data.transpose()
print(data.shape)

data = data.fillna(0)

global_max = data.max().max()
global_min = data.min().min()

print(global_max)
'''
centroid = clf.cluster_centers_

num_clusters = 45

fig = plt.figure()
for i in range(0,num_clusters):
    #plt.xlim(0, global_max)
    #plt.ylim(0, global_max)
    #plt.xlabel(str(list(data)[i]))
    #plt.ylabel(str(list(data)[j]))
    centroid = np.interp(centroid, (centroid.min(), centroid.max()), (0, +1))
    #test = np.reshape(centroid, (bins,bins))
    #plt.imshow(test)
    plt.scatter(centroid[i][:285], centroid[i][285:], s=2, alpha=0.4)
    fig.savefig( 'mood_disorders_' + str(num_clusters) + '.png')

plt.close()



