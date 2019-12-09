import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from random import randint
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.externals import joblib
from sklearn import metrics


import numpy as np
import pandas as pd
from random import randint


data = pd.read_csv('MoodDisorder_GEM.txt',sep='\t', header = 0, index_col=0)
data = data.fillna(0)
data = data.transpose()

print(data.shape)

print(len(list(data.index)))

global_max = data.max().max()
global_min = data.min().min()

print(global_min)
print(global_max)

tdata = []
f = open('MoodDisoder_edgelists.txt', 'w')

for l in range(0, 200000):
    #if l % 1000 == 0:
    #    print(l)
    i = randint(0, 66992)
    j = randint(0, 66992)
    if i != j:
    	g1 = str(data.ix[:, i].name)
    	g2 = str(data.ix[:, j].name)
    	if data.ix[:, i].max() > 0:
            if data.ix[:, j].max() > 0:
            	gene_data = data.ix[:, i].tolist() + data.ix[:, j].tolist()
            	gene_data = np.asarray(gene_data)
            	# gene_pair = np.asarray(gene_data)
            	np.nan_to_num(gene_data)
            	gene_data[gene_data < 0] = 0
            	gene_data = np.ndarray.tolist(gene_data)
            	tdata.append(gene_data)
	    	    f.write(g1)
            	f.write('\t')
            	f.write(g2)
            	f.write('\n')
f.close()
print("Finished prep")

np.savetxt('mood_disorder_bin.out', tdata, delimiter='\t')

#num_clusters = 5

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("1")
#num_clusters = 10

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("2")
#num_clusters = 15

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("3")
#num_clusters = 20

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("4")
#num_clusters = 25

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("5")
num_clusters = 30

#tdata = pd.read_csv('mood_disorder_bin.out',sep='\t')
kmeans = KMeans(n_clusters=num_clusters)
kmeans = kmeans.fit(tdata)
labels = kmeans.labels_
clusters = kmeans.predict(tdata)
tdata = np.asarray(tdata)
print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))
np.savetxt("labels_mood.txt", clusters)

#print("6")
#num_clusters = 35

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("7")
#num_clusters = 40

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("8")
#num_clusters = 45

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))

#print("9")
#num_clusters = 50

#kmeans = KMeans(n_clusters=num_clusters)
#kmeans = kmeans.fit(tdata)
#labels = kmeans.labels_
#tdata = np.asarray(tdata)
#print(metrics.silhouette_score(tdata, labels, metric='euclidean'))
#joblib.dump(kmeans, 'Mood_disorder_' + str(num_clusters))


