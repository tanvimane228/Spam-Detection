#Spam detection

#Importing the libraries
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('spam.csv')
x=dataset["EmailText"]
y=dataset["Label"]

#Splittiing the dataset into the training set and test set
x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

#Extracting features
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
features =cv.fit_transform(x_train)

#Fitting SVR to the dataset
from sklearn.svm import SVC
model=SVC(kernel = 'linear', random_state = 0)
model.fit(features,y_train)

#Test Accuracy
print(model.score(cv.transform(x_test),y_test))
