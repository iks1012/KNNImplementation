
##This is an implementation of the KNN Classifier showing 90%+ accuracy at identifying flower petals


import sys
class KNNIshanVersion:
	def fit(self, features, label):
		self.data = features
		self.labels = label
	def predict(self, test_data):
		answers = []
		for point in test_data:
			index = 0
			min_dist = n_dimensional_euclid_dist(self.data[0], point)
			for i in range(len(self.data)):
				temp_dist = n_dimensional_euclid_dist(self.data[i], point)
				if temp_dist < min_dist:
					index = self.labels[i]
					min_dist = temp_dist
			answers.append(index)
		return answers

def n_dimensional_euclid_dist(point1, point2):
		dist = 0.0
		for i in range(len(point1)):
			dist += ((point1[i] - point2[i])**2)
		return (dist**0.5) # sqrt


from sklearn import datasets
iris = datasets.load_iris()

X = iris.data;
y = iris.target;



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)


#from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNNIshanVersion() # was KNeighborsClassifier()


my_classifier.fit(X_train, y_train)


predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


