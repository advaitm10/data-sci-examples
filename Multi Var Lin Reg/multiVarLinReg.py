import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data from CSV (Vars to use: y= Wage, x= Overall, Potential, Age)
data= pd.read_csv('../fifa19.csv')
y= np.asarray(data['Wage'])
m= np.shape(y)[0]
for i in range(m):
    try:
        y[i]= float(y[i][1:len(y[i])-1])
    except:
        y[i]= float(y[i][1:])
x= np.ones((m, 4), dtype= 'float')
h= np.array([80, 0, 0, 0])
x[:, 1]= np.asarray(data['Overall'], dtype= 'float')
x[:, 2]= np.asarray(data['Potential'], dtype= 'float')
x[:, 3]= np.asarray(data['Age'], dtype= 'float')

#Feature Scaling
x[:, 1]= (x[:, 1]-70)/24
x[:, 2]= (x[:, 2]-71)/23
x[:, 3]= (x[:, 3]-30.5)/14.5

#Cost Function
def costFunction(h, x, y):
    return (1/(2*m))*sum([(x[row].dot(h)- y[row])**2 for row in range(m)])

#Gradiant Descent
alpha= 0.005 #Alpha value can prolly be optimized but idc
def gradDescent(hypothesis, x, y):
    temp= np.zeros(4)
    for i in range(0, hypothesis.size):
        temp[i]= (hypothesis[i]- (alpha/m)*sum([x[row, i]*(x[row].dot(hypothesis)-y[row]) for row in range(m)]))
    return temp

#Main
print(h, costFunction(h, x, y))
cost= costFunction(h, x, y)
for _ in range(1000):
    h= gradDescent(h, x, y)
    temp= costFunction(h, x, y)
    if(cost< temp):
        print(_)
        print(cost, temp)
        break
    else:
        cost= temp
print(h, cost)