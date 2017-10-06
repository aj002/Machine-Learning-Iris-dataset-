#Loading the iris dataset
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

#Partition the dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.5)

models = []

#Importing Logistic Regression
from sklearn.linear_model import LogisticRegression
models.append(("Logistic Regression",LogisticRegression()))

#Importing Linear Discriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
models.append(("Linear Discriminat Analysis",LinearDiscriminantAnalysis()))

#Importing K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
models.append(("KNeighborsClassifier",KNeighborsClassifier()))

#Importing Decision Tree
from sklearn.tree import DecisionTreeClassifier
models.append(("Decision Tree Classifier",DecisionTreeClassifier()))

#Import Naive Bayes
from sklearn.naive_bayes import GaussianNB
models.append(("Gaussian NB",GaussianNB()))

#Import SVC
from sklearn.svm import SVC
models.append(("SVC",SVC()))

from sklearn.metrics import accuracy_score
#Evaluating each classifier
results = []
names = []
for name, model in models:
	clf = model
	clf.fit(x_train,y_train)
	plf = clf.predict(x_test)
	print(name,":",100*accuracy_score(y_test,plf))