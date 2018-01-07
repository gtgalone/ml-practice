import pandas
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pandas.read_csv(url, names=names)

data = dataset[["sepal-length", "sepal-width", "petal-length", "petal-width"]]
label = dataset["class"]

train_data, test_data, train_label, test_label =\
  train_test_split(data, label)

clf = svm.SVC()
clf.fit(train_data, train_label)
results = clf.predict(test_data)

print(data)
print()
print(train_data)
print()
print(test_data)
print()
print(test_label)
print()
print(results)

score = metrics.accuracy_score(test_label, results)
print("정답률:", score)