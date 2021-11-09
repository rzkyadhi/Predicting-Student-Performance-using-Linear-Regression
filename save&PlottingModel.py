import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    #Declare Linear Regression Method
    linear = linear_model.LinearRegression()

    #Declare the models
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#Getting the value of m and b from the equation y = mx + b
print('Coefficient', linear.coef_)      #Getting the value of m or slope
print('intercept', linear.intercept_)   #Getting the value of b or intercept

#Getting the predictions
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#Plotting the data
p1 = "G1"
p2 = "G2"
style.use("ggplot")
pyplot.scatter(data[p2], data["G3"])
pyplot.xlabel(p1)
pyplot.ylabel("Final Grade")
pyplot.show()



