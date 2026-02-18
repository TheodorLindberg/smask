import numpy as np



class MajorityClassifier:
    def fit(self, y):
        # Find most frequent class
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
    
    def predict(self, X):
        # Predict majority class for all samples
        return np.full(len(X), self.majority_class)
    
#majority = MajorityClassifier()
#majority.fit(y_train)
#majority_predict = majority.predict(X_test)

from sklearn.dummy import DummyClassifier
class MajorityClassifierSklearn:
    def __init__(self):
        self.classifier = DummyClassifier(strategy="most_frequent")
    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)