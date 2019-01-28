import  numpy as np
import matplotlib as plt
import seaborn as sbs
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import pandas as pd

##load data
bank = pd.read_csv("M:\\ce888labs\\labs\\lab3\\lab3\\bank-additional-full.csv")
print(list(bank.columns.values))

mlb = MultiLabelBinarizer()
bank = bank.join(pd.DataFrame(mlb.fit_transform(bank.pop("y")),
                          columns=mlb.classes_,
                          index=bank.index))

##deleteing y_no and duration (del df_copy["attrbute"])
del bank['duration']
##y is split up into y yes and y no --> one hot encode these
del bank['no']


##create classifier
clf = svm.SVC(gamma=0.0001, C=100.)
k_fold = KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(bank.data[:-4000]):#has ~41k examples
    print('Train: %s | test: %s' % (train_indices, test_indices))
    clf.fit(bank.data[train_indices], bank.target[train_indices])
    print('Fold test accuracy: {} %'.format(clf.score(bank.data[test_indices], bank.target[test_indices])*100))


##convert to dummies



##plot histogram of label y_yes
plt.hist(bank['yes'])
plt.ylabel(bank['yes'])
plt.show()
##report results of 10-kfold

##get feature importanace and confusion matrix
