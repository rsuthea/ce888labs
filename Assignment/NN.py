
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Sequential
from PreProcessing import a,b,c

def selectFeatures(a):
    #this will be a final dataset for
    return N_features

Dataset1=selectFeatures(a)
Dataset2=selectFeatures(b)
Dataset3=selectFeatures(c)

#number of neurons will be changed later depending on size of datasets
N=0
model = Sequential()
model.add(Dense(N))
model.add(Dense(N))

#final NN will be built later
#keras includes all functions that are needed to train, test, etc. so not need for implementation