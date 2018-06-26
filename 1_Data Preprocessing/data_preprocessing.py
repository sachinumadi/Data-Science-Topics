# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
X
Y = dataset.iloc[:, 3].values
Y

#Taking Care of Missing Data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])

onehotEncoder = OneHotEncoder(categorical_features=[0])
X = onehotEncoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
Y= labelEncoder_y.fit_transform(Y)

#Splitting the dataset into Training Set and Testing Set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)














