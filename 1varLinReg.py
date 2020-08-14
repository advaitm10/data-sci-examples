import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data from CSV (Just make it height vs heading accuracy)
data= pd.read_csv('data.csv')
data= data.drop([x for x in range(13236, 13284)])
y= np.asarray(data['HeadingAccuracy'], dtype= 'double')
m= np.shape(y)[0]
x= np.ones((m, 2), dtype= 'double')
h= np.zeros(2)
temp= np.asarray(data['Height'])
for i in range(m):
    x[i, 1]= int(temp[i][0])*12 + int(temp[i][2])


# Cost Function
def costFunction(hypothesis, x, y, m):
    return (1/(2*m))*sum([(hypothesis*x[row]-y[row])**2 for row in range(m)])

# Gradiant Descent
alpha= 1
def gradDescent(hypothesis, x, y):
    temp= np.zeros(2)
    temp[0]= (hypothesis[0]- (alpha/m)*sum([x[row].dot(hypothesis)-y[row] for row in range(m)]))
    for i in range(1, hypothesis.size):
        temp[i]= (hypothesis[i]- (alpha/m)*sum([x[row, 1]*(x[row].dot(hypothesis)-y[row]) for row in range(m)])) #This line returns a vector rather than a value
    return temp

# Main
h= gradDescent(h, x, y)

#Plot
plt.plot(,,'ro')
