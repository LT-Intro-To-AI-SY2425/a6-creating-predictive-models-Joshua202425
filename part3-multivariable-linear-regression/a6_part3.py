import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)
#create linear regression model
rm = LinearRegression().fit(xtrain, ytrain)
#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef = np.around(rm.coef_, 2)
int= round(float(rm.intercept_), 2)
r = round(rm.score(x, y),2)
print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {int}")

print("R Squared value:", r)

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")
pred= rm.predict(xtest)
pred= np.around(pred, 2)
print(pred)
for index in range(len(xtest)):
    actual = ytest[index] 
    predicted_y = pred[index] 
    x_coord = xtest[index] 
    print(f"Car miles: {x_coord[0]} Car age: {x_coord[1]} Actual: {actual} Predicted: {predicted_y}")
