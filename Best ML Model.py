# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score

print("Multiple Linear Regression: " + str(r2_score(y_test, y_pred)))

# Polynomial
# Training the Polynomial Regression model on the Training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

# Predicting the Test set results
y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score

print("Polynomial Regression: " + str(r2_score(y_test, y_pred)))

# SVR
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X_FeatureTrain = sc_X.fit_transform(X_train)
y_FeatureTrain = sc_y.fit_transform(y_train.reshape(-1, 1))

# Training the SVR model on the Training set
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X_FeatureTrain, y_FeatureTrain.ravel())

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1, 1))
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score

print("Support Vector Regression: " + str(r2_score(y_test, y_pred)))

# Decision Tree
# Training the Decision Tree Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score

print("Decision Tree Regression: " + str(r2_score(y_test, y_pred)))

# Random Forest Regression
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Evaluating the Model Performance
from sklearn.metrics import r2_score

print("Random Forest Regression: " + str(r2_score(y_test, y_pred)))