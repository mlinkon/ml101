#Basic pipeline to predict data using KNN and Decision Tree Algorithm
#create class ScrappyKNN for implementation of algorithm 
#comments needs to be updated

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions=[]
        for row in X_test:
            label=self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self,row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row,self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = 1
        return self.y_train[best_index]     

#calculation of euclidean distance, similar to pythagoros theorm for calcuation spatial ditance betweenfeatures and labels?
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

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

# used ScrappyKNN algorithm for classification of data

kNeighborsClassifierObject = ScrappyKNN()

kNeighborsClassifierObject.fit(X_train,y_train)
kNeighborsPredictions = kNeighborsClassifierObject.predict(X_test)
print(kNeighborsPredictions)

#To calculate accuracy, compare the predicted data with test data true labels
from sklearn.metrics import accuracy_score
print('KNN ACCURACY :',accuracy_score(y_test, kNeighborsPredictions))