import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Student_Marks.csv")
print(data.head(5))




#mean squared error function
def loss_function(m,b, points):
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].time_study
        y = points.iloc[i].Marks

        total_error += (y - (m*x + b))**2

    mean_squared_error = total_error/float(len(points))

#gradients
def gradient_descent(m_now,b_now,points,L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].time_study
        y = points.iloc[i].Marks

        m_gradient += -(2/n)*x*(y - (m_now*x + b_now))
        b_gradient += -(2/n)*(y - (m_now*x + b_now))

    m = m_now - L*m_gradient
    b = b_now - L*b_gradient

    return m,b

m = 0
b = 0
epochs = 300

L=0.0001

for i in range(epochs):
    if i%50==0:
        print(f"Epochs : {i}")
    m ,b = gradient_descent(m,b,data,L)

print(m,b)

plt.scatter(data.time_study,data.Marks,color="black")
plt.plot(list(range(1,8)),[m*x+b for x in range(1,8)],color="red")
plt.show()









