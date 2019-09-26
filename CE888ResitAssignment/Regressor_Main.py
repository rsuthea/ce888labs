#TODO:
#get 3 multivariate regression datasets
#run linear regression with high l1 regularisation - record the results using 10 fold cross validation
#desicion tree regressor with maximum depth of 3 - record the results using 10 fold cross validation
#compare results


#https://www.youtube.com/watch?v=gJo0uNL-5Qw


import numpy as np
from scipy.interpolate import *
import matplotlib.pyplot as plt
from IPython.display import Image
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz
from graphviz import Source
import sklearn
from sklearn.model_selection import train_test_split, KFold, validation_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree, linear_model
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix
from subprocess import check_call
import pydotplus
from seaborn import heatmap
from graphviz import Source

def printTree(clf_tree):
    import os
    os.environ["PATH"] += os.pathsep + ';C:/Program Files (x86)/Graphviz2.38/bin/'
    #only works for the last dataset in the array, change to what is needed
    tree.export_graphviz(clf_tree,out_file='tree.dot')

    dot_data = StringIO()
    tree.export_graphviz(clf_tree, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #s=Source.from_file('tree.dot')
    #s.view()
    graph.write_png("dtree.png")

def traintestsplit(dataset):
    # todo: these must change to fit what dataset it is - all use 3 columns
    x = dataSet.iloc[:, :-2]
    y = dataSet.iloc[:, -2:]

    # 0,1 squishes down everything, may not be good to keep
    scalerX = MinMaxScaler()
    x = scalerX.fit_transform(x)
    scalerY = MinMaxScaler()
    y = scalerY.fit_transform(y)

    return x,y

def testgridsearchcv(data):
    regmodel=[]
    dtmodel=[]

    x,y=traintestsplit(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    #set random state to 1 to stop random seeding
    desTree = DecisionTreeRegressor(random_state=0)
    param_grid = {'max_depth': range(1,3), 'min_samples_split' : range(10,500,20)}
    clf = sklearn.model_selection.GridSearchCV(estimator=desTree, param_grid=param_grid, verbose=0, cv=10, scoring='r2')
    clf=clf.fit(x_train,y_train)
    print('test',clf.best_score_)
    predvals = clf.predict(x_test)
    plt.plot(predvals)
    plt.show()

    # printTree(clf)

    lasso = linear_model.Lasso()  # alpha=x went here
    lasso_param = {'alpha': range(0.001, 0.01, 0.1)}
    clf = sklearn.model_selection.GridSearchCV(lasso, lasso_param, verbose=0, cv=10)
    clf = clf.fit(x_train, y_train)
    print('test', clf.best_score_)
    predvals = clf.predict(x_test)
    plt.plot(predvals)
    plt.show()

    sgdmodel = linear_model.SGDRegressor()
    param_grid = {'max_iter': range(100, 1000), 'tol': range(0.001,0.01,0.1)}
    clf = sklearn.model_selection.GridSearchCV(estimator=sgdmodel, param_grid=param_grid, verbose=0, cv=10, scoring='r2')
    clf = clf.fit(x_train, y_train)
    print('test', clf.best_score_)
    predvals = clf.predict(x_test)
    plt.plot(predvals)
    plt.show()

#fitting and testing each model
def model(model,xtrain,xtest,ytrain,ytest):
    
    clf = model.fit(xtrain,ytrain)
    #todo: remove this  :  plot_confusion_matrix(xtest, ytest)
    result =clf.score(xtest,ytest)
    pred=clf.predict(xtest)
    i=0
    tester=[]
    """
    while i<np.size(result):
        tester.append([np.mean(ytest[0][i]),np.mean(pred[0][i])])

        i+=1
    print(np.size(tester))
    cm = confusion_matrix(tester[0],tester[1])
    heatmap(cm)
     """
    return result


#more tradition kfold
def K_Fold(dataSet,dataSetName):
    regmodel=[]
    dtmodel=[]
    sgdmodel=[]

    x,y=traintestsplit(dataSet)

    kf = KFold(n_splits=10, shuffle=False)
    for trainData, testData in kf.split(x,y):
        #print("Train Index: ", trainData, "\n")
        #print("Test Index: ", testData)
        xtrain, xtest = x[trainData], x[testData]
        ytrain, ytest = y[trainData], y[testData]

        #decision tree learning
        dtmodel.append(model(tree.DecisionTreeRegressor(max_depth=3, random_state=0),xtrain,xtest,ytrain,ytest)) # change max depth as necessary

        #SGDRegressor learning
        ytrainsgd=np.argmax(ytrain,axis=-1)
        ytestsgd = np.argmax(ytest, axis=-1)
        sgdmodel.append(model(linear_model.SGDRegressor(max_iter=1000,tol=0.01),xtrain,xtest,ytrainsgd,ytestsgd))

        #linear model learning (lasso)
        lasso=linear_model.Lasso()#alpha=x went here
        lasso_param ={'alpha':[0.001,0.01,0.1]}
        lasso_grid= sklearn.model_selection.GridSearchCV(lasso, lasso_param, verbose=0, cv=10)
        regmodel.append(model(lasso_grid, xtrain, xtest, ytrain, ytest))

        #regmodel.append(model(lasso, xtrain, xtest, ytrain, ytest))

    print("dt model scores= ", dtmodel)
    print("regmodel scores= ", regmodel)
    print("sgdmodel scores= ", sgdmodel)
    print("dt model avg = ", np.mean(dtmodel))
    print("regmodel avg = ",np.mean(regmodel))
    print("sgdmodel avg = ",np.mean(sgdmodel))
    plt.figure(dataSetName)
    plt.title("lasso vs decision tree")
    plt.xlabel("fold")
    plt.ylabel("score")
    plt.plot(dtmodel, "x")
    plt.plot(regmodel, 'o')
    filesave=str(dataSetName+ ".png")
    plt.savefig(filesave)
    #print(confusion_matrix(ytest, model.predict(xtest)))



if __name__=="__main__":
    #datasets:
    ##flare data : https://archive.ics.uci.edu/ml/datasets/Solar+Flare
    ##air qulaity : https://archive.ics.uci.edu/ml/datasets/Air+Quality
    #fb one contains NaN or infinity values - https://archive.ics.uci.edu/ml/datasets/Facebook+metrics#

    datasetdir={"AirQuality.csv","FlareData.csv","DatasetFacebook.csv"}
    for i in datasetdir:
        dataSet = pd.read_csv(i)
        dataSetName=str(i)
        ##comment either out as needed
        K_Fold(dataSet,i)
        #testgridsearchcv(i)
        