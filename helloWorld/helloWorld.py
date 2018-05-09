from sklearn import tree
features = [[140,1],[130,1],[160,0],[170,0]]  #[weight, texture], texture type : 1 is for smooth, 0 is for bumpy
labels = ["apple","apple","orange","orange"]  #features are input and labels are output.
clf = tree.DecisionTreeClassifier()           #using decision tree for data classification
clf = clf.fit(features, labels)               #fit is synonym for find pattern in data
print (clf.predict([[100,0]]))