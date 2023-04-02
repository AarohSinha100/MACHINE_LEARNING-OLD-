import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#creating the dataset
area = [2600,3000,3200,3600,4000]
price = [550000,565000,610000,680000,725000]
dataset = [[2600,550000],[3000,565000],[3200,610000],[3600,680000],[4000,725000]]

df = pd.DataFrame(dataset,columns=['Area','Price'])

df.to_csv('area_price')
##########################


data = pd.read_csv('area_price')


#selecting the x and y values
X = data['Area'].values
Y = data['Price'].values

main = plt.scatter(X,Y)
#plt.show()

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)

numer = 0
denom = 0

for i in range(n):
    numer += (X[i]-mean_x)*(Y[i]-mean_y)
    denom += (X[i]-mean_x)**2

m = numer/denom
c = mean_y - (m*mean_x)

print(m,c)

x = X
y = []

for i in range(len(x)):
    n = m*x[i]+c
    y.append(n)


#checking by Rsquared method



ss_t = 0
ss_r = 0

for i in range(n):
    ss_t = (y_pred[i] - mean_y)**2
    ss_r = (y[i] - mean_y)**2
    
r2 = ss_t/ss_r
print(r2)

#Checking the graph
plt.scatter(data.Area,data.Price)
plt.plot(x,y)
plt.show()

      





