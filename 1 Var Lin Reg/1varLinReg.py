import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data from CSV
data= pd.read_csv('../fifa19.csv')
y= np.asarray(data['Overall'], dtype= 'float')
m= np.shape(y)[0]
x= np.ones((m, 2), dtype= 'float')
h= np.array([80, 0])
temp= np.asarray(data['Wage'])
for i in range(m):
    try:
        x[i, 1]= float(temp[i][1:len(temp[i])-1])
    except:
        x[i, 1]= float(temp[i][1:])

# Cost Function
def costFunction(hypothesis, x, y, m):
    return (1/(2*m))*sum([(x[row].dot(hypothesis)-y[row])**2 for row in range(m)])

# Gradiant Descent
alpha= 0.003
def gradDescent(hypothesis, x, y):
    temp= np.zeros(2)
    temp[0]= (hypothesis[0]- (alpha/m)*sum([x[row].dot(hypothesis)-y[row] for row in range(m)]))
    for i in range(1, hypothesis.size):
        temp[i]= (hypothesis[i]- (alpha/m)*sum([x[row, 1]*(x[row].dot(hypothesis)-y[row]) for row in range(m)]))
    return temp

# Main
print(h, costFunction(h, x, y, m))
cost= costFunction(h, x, y, m)
for _ in range(1000):
    h= gradDescent(h, x, y)
    temp= costFunction(h, x, y, m)
    if(cost< temp):
        print(_)
        print(cost, temp)
        break
    else:
        cost= temp
print(cost)

#Plot
outStr= '{:.2f}x + {:.2f}'
plt.plot(x[:, 1], y, 'ro', label= 'Data')
plt.plot([0, 600],[h[0], h[0]+(h[1]*600)],label= outStr.format(float(h[1]), float(h[0])))
plt.axis([0, 600, 42, 99])
plt.legend()
plt.show()