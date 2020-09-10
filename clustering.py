#!/usr/bin/python3
import cgi, os
import csv
import cgitb; cgitb.enable()
print('Content-type: text/html\r\n')
form = cgi.FieldStorage()
# Get filename here.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#
from sklearn.cluster import KMeans
from sklearn import metrics





fileitem = form['filename']
clusterSize = form['clusterSize']
# Test if the file was uploaded
if fileitem.filename:
   # strip leading path from file name to avoid
   # directory traversal attacks
   command=form.getvalue("cmd")
   clusterSize = form.getvalue('clusterSize')

   fn = os.path.basename(fileitem.filename.replace("\\", "/" ))
   open('/tmp/' + fn, 'wb').write(fileitem.file.read())

   message = 'The file "' + fn + '" was uploaded successfully'
   # seems like nothing can go here

else:
   message = 'No file was uploaded'

print("<h1> Clustering Output </h1>")

print('<a href="http://yoda.kean.edu/~niedzwid/CPS4721/hw3.2.html">Upload another file</a>')


print('<br>')

file = '/tmp/' + fn

with open( file ) as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames



data = pd.read_csv(file)




#initali cluster plot
x=data.copy()
kmeans=KMeans( int(clusterSize) )
kmeans.fit(x)
clusters=x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)
plt.scatter(clusters[ headers[1] ], clusters[ headers[2] ], c=clusters['cluster_pred'], cmap='rainbow')




##centriod detection
plt.plot()
X = np.array(list(zip( clusters[ headers[1] ] , clusters[ headers[2] ] ))).reshape(len( clusters[headers[1]] ), 2)
K = int(clusterSize)
kmeans_model = KMeans(n_clusters=K).fit(X)
centers = np.array(kmeans_model.cluster_centers_)
plt.plot()
plt.scatter(centers[:,0], centers[:,1], marker="x", color= 'black')





plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.savefig('../test3.png')
print("<img src='../test3.png'>")



print(data.to_html())
