#!/usr/bin/python3
import os
import cgi

print('Content-type: text/html\r\n')
# form=cgi.FieldStorage()
# command=form.getvalue("cmd")

import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt,mpld3

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist





data = pd.read_csv('data.csv')
print('Input Data and Shape')
print(data.shape)
data.head()
#getting the actual data seperated

f1 = data['X'].values
f2 = data['Y'].values

x=np.array(list(zip(f1,f2)))

k = 2

C_x = np.random.randint(0 , np.max(x)-20, size=k)

C_y = np.random.randint(0 , np.max(x)-20, size=k)

C=np.array(list(zip(C_x, C_y)), dtype=np.float32)

fig1 = plt.figure()

plt.scatter(f1,f2, c="#050505", s=20)
plt.scatter(C_x, C_y, marker = '*', s=200, c='g')

plt.xlabel('distance')
plt.ylabel('speed')
plt.title('Cluster')

print ('<H3>Graph</H3>')
print (mpld3.fig_to_html(fig1, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None, use_http=False))
print ('<br>')




kmeans= KMeans(n_clusters=k)

kmeans=kmeans.fit(x)

labels = kmeans.predict(x)

centroids = kmeans.cluster_centers_

print(centroids)
