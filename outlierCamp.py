#!/usr/bin/python3
import os
import cgi


print('Content-type: text/html\r\n')
form=cgi.FieldStorage()
command=form.getvalue("cmd")
#
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#
from sklearn.cluster import KMeans

from sklearn import metrics


data = pd.read_csv('data.csv')

print(data)


plt.scatter(data['X'], data['Y'])
plt.xlabel('x')
plt.xlabel('y')

# plt.show()
#

x=data.copy()
kmeans=KMeans(4)
kmeans.fit(x)

clusters=x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)

plt.scatter(clusters['X'], clusters['Y'], c=clusters['cluster_pred'], cmap='rainbow')





plt.plot()
X = np.array(list(zip(clusters['X'], clusters['Y']))).reshape(len(clusters['X']), 2)


K = 4
kmeans_model = KMeans(n_clusters=K).fit(X)

# print(kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)

plt.plot()
plt.scatter(centers[:,0], centers[:,1], marker="x", color= 'black')





plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.savefig('../test3.png')

print("<img src='../test3.png'>")











# https://saskeli.github.io/data-analysis-with-python-summer-2019/clustering.html
