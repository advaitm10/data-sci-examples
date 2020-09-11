import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Import data from CSV (Maybe do GKDiving to predict whether someone is a GK)
data= pd.read_csv('../fifa19.csv')
y= np.asarray(data['Position'])
m= np.size(y)[0]
x= np.ones((m, 2), dtype= 'Float')
x= np.asarray(data['GKDiving'], dtype= 'float')
h= np.array([0, 0])
for i in range(m):
    if(y[i]== 'GK'):
        y[i]= 1
    else:
        y[i]= 0

#Sigmoid Function
def sigmoid(z): #consider switching to numpy.exp
    return 1/(1+ math.exp((-1*z)))

#Cost Function
def cost(h, m, x, y):
    return sum([y[row]*math.log10(sigmoid(h.dot(x[row]))) + (1-y[row])*math.log10(1-sigmoid(h.dot(row[x]))) for row in range(m)])

#Gradient Descent
alpha = 0.1
def gradDescent(h, m, x, y, alpha): #Finish this
    return h - ((alpha/m) * (x.dot(sigmoid(h.dot(x)- y))))