#usign Decision Tree for classification of vehicles based on horsepower and number of seats 
from sklearn import tree
features = [[300,2],[450,2],[200,8],[150,9],[250,4],[300,4]]                            #[horsepower, number of seats in vehicle]
labels = ["sports-car","sports-car","minivan","minivan","utility-car","utility-car"]    #features are input and labels are output.
clf = tree.DecisionTreeClassifier()                                                     #using decision tree for data classification     
clf = clf.fit(features, labels)                                                         #fit is synonym for find pattern in data
print (clf.predict([[275,2]]))
print (clf.predict([[150,4]]))
print (clf.predict([[200,6]]))