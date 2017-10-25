# svm.py
# -------------

# svm implementation
import util
from sklearn import svm
# from sklearn.metrics import accuracy_score
# from dataClassifier import getNumpyData

PRINT = True

class SVMClassifier:
  """
  svm classifier
  """
  def __init__( self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "svm"
    self.clf = svm.LinearSVC()
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    X = trainingData
    y = trainingLabels
    self.clf.fit(X,y)

    
  def classify(self, data ):
    guesses = self.clf.predict(data)
      
    return guesses

if __name__ == '__main__':
  pass
    # train, val, test, trainLabels, valLabels, testLabels = getNumpyData(2000,500)
    # clf = svm.LinearSVC()
    # clf.fit(train, trainLabels)
    # pred = clf.predict(test)
    # accuracy_score(testLabels, pred)



