#Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('tvmarketing.csv')
df.head()
df.info()
df.shape
df.describe()
plt.figure(figsize=(10,5))
plt.scatter(df['TV'], df['Sales'])
plt.title("TV marketing budget Vs Sales")
plt.xlabel('TV Marketing Budget')
plt.ylabel('Sales')
plt.show()
X=df['TV']
y=df['Sales']
print(X.shape, y.shape)
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)

print(X_train.shape,X_test.shape, y_train.shape,y_test.shape)
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
print(X_train.shape,X_test.shape, y_train.shape,y_test.shape)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
# Print the intercept and coefficients
print(lr.intercept_)
print(lr.coef_)
slope = lr.coef_[0]
intercept = lr.intercept_[0]
# Plot the regression line in the scatter plot between TV and Sales values.
plt.style.use('dark_background')
plt.figure(figsize = (12, 4), dpi = 96)
plt.title("Regression Line", fontsize = 16)
plt.scatter(df['TV'], df['Sales'])
plt.plot(df['TV'], slope * df['TV'] + intercept, color = 'r', linewidth = 2, label = '$y = 7.23x + 0.04$')
plt.xlabel("TV")
plt.ylabel("Sales")
plt.legend()
plt.show()
# Making predictions on the testing set
y_pred = lr.predict(X_test)

y_pred[:5]
x_axis = [i for i in range(1,61)]
plt.plot(x_axis,y_test, color="blue", linewidth=2, linestyle="-")
plt.plot(x_axis,y_pred, color="red",  linewidth=2, linestyle="-")
plt.title('Actual Vs Predicted')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.show()
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
pred = lr.predict([[1000]])
pred
