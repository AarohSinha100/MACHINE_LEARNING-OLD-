import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
####print(data)
#loss function
def loss_function(m,b,points):
    total_error = 0 #we add all the individual squared errors to it and then we divide by n
    for i in range(len(points)):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Marks
        total_error += (y - (m*x+b))**2
    total_error / float(len(points))

#L - learning rate
def gradient_descent(m_now,b_now,points,L): #derivatives
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Marks

        m_gradient+= -(2/n)*x*(y-(m_now*x+b_now))
        b_gradient+= -(2/n)*x*(y-(m_now*x+b_now))

    m = m_now - m_gradient*L
    b = b_now - b_gradient*L

    return m,b

m = 0
b = 0
L = 0.0001
epochs = 300

for i in range(epochs):
    if i%50==0:
        print(f"Epochs {i}")
    m,b = gradient_descent(m,b,data,L)

print(m,b)
plt.scatter(data.Study_Time,data.Marks,color="black")
plt.plot(list(range(3,100)),[m*x+b for x in range(3,100)],color="red")
plt.show()






