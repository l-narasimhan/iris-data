# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:15:21 2017

@author: LaN
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

#Read data from Iris.csv
data = pd.read_csv('iris-species\Iris.csv')

print data.head()
print data.describe()

# Map the labels
data['Species'] = data['Species'].map({'Iris-setosa':1,'Iris-versicolor':2, 'Iris-virginica':3})
#print data.head()


x= data['SepalWidthCm']
y = data['Species']
#plt.scatter(x,y, color = 'g')
#plt.legend()

y = data['PetalLengthCm']
x = data['Species']
#plt.scatter(x,y, color = 'b')

x = data['PetalWidthCm']
y = data['Species']
#plt.scatter(x,y, color = 'b')

features = data.ix[0::,1:5]
labels = data.ix[0::,5]

# Remove Id and Species columns from Training data
f1 = list(data.columns)
f1.remove('Species')
f1.remove('Id')

# Scale the features

features[f1] = features[f1].apply(lambda x: x/x.max(), axis=0)
#print features
"""
x1 = features.ix[::,0]
print x1
x2 = features.ix[::,1]
print x2
x3 = features.ix[::,2]

x4 = features.ix[::,3]

plt.scatter(x1,x2, color= 'b''r')

plt.scatter(x3,x4, color = 'g''r')
"""

from sklearn import svm
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.3, random_state=0)

print('There are {} samples in the training set and {} samples in the test set'.format(
features_train.shape[0], labels_test.shape[0]))
print()

### Support Vector Machines

linear_kernel_svm = svm.SVC(kernel='rbf', C=1000)

linear_kernel_svm.fit(features_train, labels_train)

print "Support Vector Machines"

print "Accuracy on Training data :",linear_kernel_svm.score(features_train, labels_train)

pred = linear_kernel_svm.predict(features_test)

print "Accuracy on Test data:", linear_kernel_svm.score(features_test, labels_test)

print "-----------------------"
### Random Forest Classifier 

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(features_train, labels_train)

print "Random Forest Classifier"
print "Accuracy on Training data:", rf.score(features_train, labels_train)

pred = rf.predict(features_test)

print "Accuracy on Test data:", rf.score(features_test, labels_test)



