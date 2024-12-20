import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print(x)
print(y)
# Step 2: Standardize the data using StandardScaler, 
scale = StandardScaler().fit(x)
# Step 3: Transform the data
x = scale.transform(x)
# Step 4: Split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y)
# Step 5: Fit the data
# Step 6: Create a LogsiticRegression object and fit the data
m = linear_model.LogisticRegression().fit(xtrain, ytrain)
# Step 7: Print the score to see the accuracy of the model
print(m.score(xtest, ytest))
# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
for i in range(len(xtest)):
    x = xtest[i]
    x = x.reshape(-1, 3)
    ypredi = int(m.predict(x))
    if ypredi == 0:
        ypredi = "No"
    elif ypredi == 1:
        ypredi = "Yes"
    a = ytest[i]
    if a == 0:
        a = "No"
    elif a == 1:
        a = "Yes"
    print("Predicted - " + ypredi + " Actual - " + a)
    