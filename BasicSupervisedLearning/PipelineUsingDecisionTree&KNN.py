#Basic pipeline to predict data using KNN and Decision Tree Algorithm
#import a dataset
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
#f(x) = y  --function applied on data set x= features y =output decision

#spliting the dataset into two equal halves, one is used for training and other is used for testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(x,y,test_size=.5)

# X_train are features of training data y_train are labels of training data
# y_train are features of testing data, y_test are the labels of testing data

#used Decision tree for classification of data
from sklearn import tree
decisionTreeClassifier = tree.DecisionTreeClassifier()  

decisionTreeClassifier.fit(X_train,y_train) 
decisionTreePredicitions = decisionTreeClassifier.predict(X_test) #predicts the test data labels
print(decisionTreePredicitions)

# used K-Nearest Neighbors aka KNN algorithm for classification of data
from sklearn.neighbors import KNeighborsClassifier
kNeighborsClassifierObject = KNeighborsClassifier()

kNeighborsClassifierObject.fit(X_train,y_train)
kNeighborsPredictions = kNeighborsClassifierObject.predict(X_test)
print(kNeighborsPredictions)
#To calculate accuracy, compare the predicted data with test data true labels
from sklearn.metrics import accuracy_score
print('Decision tree ACCURACY : ', accuracy_score(y_test, decisionTreePredicitions))

print('KNN ACCURACY :',accuracy_score(y_test, kNeighborsPredictions))