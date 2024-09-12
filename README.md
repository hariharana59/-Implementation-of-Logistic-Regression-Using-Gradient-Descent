# Implementation-of-Logistic-Regression-Using-Gradient-Descent
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Hariharan A
RegisterNumber:  212223110013
```
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 - Start

STEP 2 -Import the necessary python packages

STEP 3 - Read the dataset.

STEP 4 - Define X and Y array.

STEP 5 - Define a function for costFunction,cost and gradient.

STEP 6- Define a function to plot the decision boundary and predict the Regression value

STEP 7- End

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Placement_Data.csv")
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['gender'] = dataset['gender'].astype('category')
dataset['ssc_b'] = dataset['ssc_b'].astype('category')
dataset['hsc_b'] = dataset['hsc_b'].astype('category')
dataset['hsc_s'] = dataset['hsc_s'].astype('category')
dataset['degree_t'] = dataset['degree_t'].astype('category')
dataset['workex'] = dataset['workex'].astype('category')
dataset['specialisation'] = dataset['specialisation'].astype('category')
dataset['status'] = dataset['status'].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset['ssc_b'] = dataset['ssc_b'].cat.codes
dataset['hsc_b'] = dataset['hsc_b'].cat.codes
dataset['hsc_s'] = dataset['hsc_s'].cat.codes
dataset['degree_t'] = dataset['degree_t'].cat.codes
dataset['workex'] = dataset['workex'].cat.codes
dataset['specialisation'] = dataset['specialisation'].cat.codes
dataset['status'] = dataset['status'].cat.codes
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y = Y 
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,Y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta = alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0,78,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print("Y_predict_new_data:",y_prednew)
```

## Output:

![image](https://github.com/user-attachments/assets/8a36f311-6721-4338-bcfe-b506bbe6dac1)

![image](https://github.com/user-attachments/assets/fef38232-f948-41e6-a56e-c5a7eb570a69)


![image](https://github.com/user-attachments/assets/2012bf96-12af-4cd0-a24b-03c3a8b6affd)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
