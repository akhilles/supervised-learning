# perceptron.py
# -------------

# Perceptron implementation
import util
import numpy as np
from dataClassifier import getNumpyData

PRINT = True


class numceptronClassifier:
    """
    Perceptron classifier.
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "numceptron"
        self.max_iterations = max_iterations
        self.weights = np.zeros((len(legalLabels), 28*28), dtype=int)

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights == weights;

    def train(self, trainingData, trainingLabels, validationData, validationLabels):

        print("Size of training data: {}".format(trainingData.shape))

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                scores = util.Counter()
                datum = trainingData[i]
                y = trainingLabels[i]
                scores = np.dot(self.weights,datum)
                best_score = np.argmax(scores)
                if (y != best_score):
                    self.weights[best_score] -= datum
                    self.weights[y] += datum

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        """
        scores = np.dot(data , self.weights.T)
        guesses = np.argmax(scores, axis= 1)
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []
        myw = self.weights[label]

        for i in range(100):
            maxindex = np.argmax(myw)
            featuresWeights.append((maxindex//28, maxindex%28))
            myw[maxindex] = -1

        return featuresWeights

if __name__ == '__main__':
    # print(getNumpyData(10,3)[1].shape)
    train, val, test, trainLabels, valLabels, testLabels = getNumpyData(2000,100)
    c = numceptronClassifier(range(10), 1)
    c.train(train,trainLabels, val, valLabels)

    guess = c.classify(val)
    print('Val Accuracy:{}'.format(1.0 * np.sum(guess == valLabels) / len(valLabels)))
    guess = c.classify(test)
    print('Test Accuracy:{}'.format(1.0*np.sum(guess==testLabels)/len(testLabels)))

