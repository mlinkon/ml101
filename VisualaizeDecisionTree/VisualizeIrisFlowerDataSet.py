#Iris Flower DataSet [Wikipedia link : https://en.wikipedia.org/wiki/Iris_flower_data_set ]
import numpy as np
import graphviz
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.feature_names) #prints list of features in dataset
print(iris.target_names)  #prints all distinct flower names in dataset
print(iris.data[0])       #prints first record in dataset. Format : [sepal length(cm) sepal Width(cm) petal length(cm) petal width(cm)] 
print(iris.target[100])   #prints the target value which is label. Output 0:setosa 1:versicolor 2:virginica
for i in range(len(iris.target)):
    print("Example {0} : label {1}, feature {2}".format(i, iris.target[i], iris.data[i]))


#Testing Data - For testing remove one example of each type of flower
#train_* variable will have majority of data and test_* variable will have only deleted data
#Step 1 : Delete data
test_index = [0,50,100]
train_target = np.delete(iris.target,test_index)
train_data = np.delete(iris.data,test_index,axis=0)

#Step 2 : Train Data
test_target = iris.target[test_index]
test_data = iris.data[test_index]


#Step 3 : Creating Decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)  #train on training data, pass features and labels in fit method 

print("expected output : {0}".format(test_target))
print ("decision tree output : {0}".format(clf.predict(test_data)))


#Visualize the decision tree
from sklearn.externals.six import StringIO
import pydotplus   #using pydotplus in windows10, python 3.6.X
dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("iris.pdf")    #outputs pdf file with decision tree diagram

#for verification of output from iris.pdf decision path
print(iris.feature_names,iris.target_names)
print(test_data[1], test_target[1])
print(test_data[2], test_target[2])
print(test_data[0], test_target[0])