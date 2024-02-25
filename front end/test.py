# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
dset = pd.read_csv("surveylungcancer.csv")
dset.info()
dset.head()
dset["GENDER"] = dset["GENDER"].map({
    'M' : '0',
    'F' : '1'
})
dset["LUNG_CANCER"]= dset["LUNG_CANCER"].map({
    'YES':'1',
    'NO':'0'
}) 
X = np.array(dset.drop("LUNG_CANCER",axis=1))
Y = np.array(dset[["LUNG_CANCER"]])
loo = LeaveOneOut()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=42)
'''
for train_index, test_index in loo.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    xtrain, xtest = X[train_index], X[test_index]
    ytrain, ytest = Y[train_index], Y[test_index]
dset.info()
'''
from sklearn import svm
model = svm.SVC()
model.fit(X,Y)

pred = model.predict(X_test)
print(y_test,pred)
acc = accuracy_score(pred, y_test)
print("Accuracy:",acc*100)