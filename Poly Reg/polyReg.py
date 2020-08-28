import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data from CSV
data= pd.read_csv('../fifa19.csv')
y= np.asarray(data['Wage'])
m= np.shape(y)[0]
for i in range(m):
    try:
        y[i]= float(y[i][1:len(y[i])-1])
    except:
        y[i]= float(y[i][1:])
h= np.array([17.69295469, -688.42725802,  817.9606341]) #Hypothesis is y= overall^2 + overall + b
x= np.ones((m, 3), dtype= 'float')
x[:, 1]= np.asarray(data['Overall'], dtype= 'float')
x[:, 2]= np.square(x[:, 1])

#Feature Scaling
x[:, 1]= (x[:, 1]-46)/48
x[:, 2]= (x[:, 2]-2116)/6720

# Cost Function
def costFunction(hypothesis, x, y):
    return (1/(2*m))*sum([(x[row].dot(hypothesis)-y[row])**2 for row in range(m)])

# Gradiant Descent
alpha= 1.5
def gradDescent(hypothesis, x, y):
    temp= np.zeros(3)
    for i in range(0, hypothesis.size):
        temp[i]= (hypothesis[i]- (alpha/m)*sum([x[row, i]*(x[row].dot(hypothesis)-y[row]) for row in range(m)]))
    return temp

# Main
print(h, costFunction(h, x, y))
cost= costFunction(h, x, y)
while True:
    h= gradDescent(h, x, y)
    temp= costFunction(h, x, y)
    print(temp)
    #Checks if cost gets higher
    if(cost< temp):
        print(cost, temp)
        break
    #Checks if cost converges
    elif(cost-temp<= 0.001):
        print('Converged')
        break
    else:
        cost= temp

print(h, cost)

#Plot
t= np.arange(-1.0, 1.0, 0.01)
s= np.asarray([h.dot([1.0, i, i**2]) for i in t])
outStr= '{:.2f}x^2 + {:.2f}x + {:.2f}'
plt.plot(x[:, 1], y, 'ro', label= 'Data')
plt.plot(t, s,label= outStr.format(float(h[2]), float(h[1]), float(h[0])))
plt.axis([0, 1, -600, 600])
plt.legend()
plt.show()