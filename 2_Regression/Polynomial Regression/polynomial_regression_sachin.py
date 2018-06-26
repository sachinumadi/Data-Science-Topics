#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to the data set
from sklearn.linear_model import LinearRegression
lin_Reg = LinearRegression()
lin_Reg.fit(X,y)


#Fitting Polynomial Linear Regression to the data set
from sklearn.preprocessing import PolynomialFeatures
ploy_reg = PolynomialFeatures(degree=4) # change degree starting from 2 to more number to get accuracy
X_poly = ploy_reg.fit_transform(X)

lin_Reg2 = LinearRegression()
lin_Reg2.fit(X_poly,y)

#Visualising Linear Regression results

plt.scatter(X,y,color='red')
plt.plot(X,lin_Reg.predict(X),color='blue')
plt.title('Truth or Bluff (linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial Linear Regression results

plt.scatter(X,y,color='red')
plt.plot(X,lin_Reg2.predict( ploy_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff (polinomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Checking for all levels from 0.1 to 10
X_grid = np.arange(min(X),max(X),0.1) 
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_Reg2.predict( ploy_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (polinomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Predicting a new result with Linear Regression
lin_Reg.predict(6.5)

#Predicting a new result with Polynomial Linear Regression
lin_Reg2.predict(ploy_reg.fit_transform(6.5))






















