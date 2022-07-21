#Importing libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression

#Importing test and train data
'''
While searching for the dataset on the internet,
I found a dataset which is already seperated into test and train data,
so i don't have to perform the test train split step.
'''
data_train = pd.read_csv("./mnist_train.csv")
data_test = pd.read_csv("./mnist_test.csv")

#Plotting image to check the data base
img = data_train.iloc[0, 1:].values
img = img.reshape(28,28).astype("uint8")
plt.imshow(img)
plt.show()

#Dividing the data into input_train data and response_train data
x_train = data_train.iloc[:, 1:]
y_train = data_train.iloc[:, 0]

#Dividing the data into input_test data and response_test data
x_test = data_test.iloc[:, 1:]
y_test = data_test.iloc[:, 0]

#Calling the model
'''
I used different models to predict the value, accuracy of each model is specified below:-
RandomForest: 96.96%
KNN (Best performance is for n_neighbors=7): 96.94%
Decisiontree: 87.64%
LogisticRegression: 92.55% (Getting a warning that total no of iterations reached the limit)
I also tried to apply SVM but it is taking infinte time to fit the dataset.
Result: RandomForest performed the best.
'''
model = RandomForestClassifier(n_estimators=100)
#model = KNeighborsClassifier(n_neighbors=7)
#model = DecisionTreeClassifier()
#model = LogisticRegression()

#Training the dataset
model.fit(x_train, y_train)

#Predicting the output for test data
pred = model.predict(x_test)

#Calculating accuracy of the model
realval = y_test.values
count = 0
predlen = len(pred)
for i in range(predlen):
    if pred[i] == realval[i]:
        count = count+1

#Printing accuracy of the model
print("Accuracy:", count*100/predlen, "%")