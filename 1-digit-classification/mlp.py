# mlp.py
# -------------

# mlp implementation
import util
import numpy as np
PRINT = True
DIGIT_SIZE = 28*28

class MLPClassifier:
  """
  mlp classifier
  """
  def __init__( self, legalLabels, max_iterations, hidden_size = 200):
    self.legalLabels = legalLabels
    self.type = "mlp"
    self.max_iterations = max_iterations
    self.hidden = hidden_size
    self.input = DIGIT_SIZE
    self.output = len(legalLabels)
    self.params = {}
    self.params['W1'] = 1e-4*np.random.rand(self.input, self.hidden)
    self.params['W2'] = 1e-4*np.random.rand(self.hidden, self.output)
    self.params['b1'] = np.zeros(self.hidden)
    self.params['b2'] = np.zeros(self.output)
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels,learning_rate=1,
            reg=1e-5, decay = 0.99, batch = True, batch_size = 200):

    X = trainingData
    y = trainingLabels
    N = X.shape[0]

    print "Starting Training..."

    for iteration in range(self.max_iterations):
      if (batch):
        ind = np.random.choice(N, batch_size)
        X_batch = X[ind]
        y_batch = y[ind]
      else:
        X_batch = X
        y_batch = y
      grads, loss = self.forward(X_batch,y_batch, reg)
      self.params['W2'] -= learning_rate*grads['W2']
      self.params['W1'] -= learning_rate*grads['W1']
      self.params['b2'] -= learning_rate*grads['b2']
      self.params['b1'] -= learning_rate*grads['b1']
      print("Iteration: {} Loss: {}".format(iteration, loss))
      if(learning_rate>0.1):
        learning_rate *= decay



  def forward(self, X, y, reg = 0.0):
    N, D = X.shape
    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1']
    b2 = self.params['b2']

    # Forward pass
    dotxw = np.dot(X, W1) + b1
    hidden_values = np.maximum(dotxw, 0)
    scores = np.dot(hidden_values, W2) + b2

    # Loss
    exps = np.exp(scores)
    sums = np.sum(exps, axis=1)
    fyi = exps[np.arange(N), y]
    costs = -1 * np.log(fyi / sums)
    loss = np.sum(costs) / N + reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)

    #compute gradients
    grads = {}

    dscores = exps / np.reshape(sums, (sums.shape[0], 1))
    dscores[range(N), y] -= 1
    dscores = dscores / N

    grads['b2'] = np.sum(dscores, axis=0)
    grads['W2'] = np.dot(hidden_values.T, dscores)
    grads['W2'] += 2 * W2 * reg

    dHidden = np.dot(dscores, W2.T)
    ddotxw = dHidden * (dotxw > 0)
    grads['b1'] = np.sum(ddotxw, axis=0)
    grads['W1'] = np.dot(X.T, ddotxw)
    grads['W1'] += 2 * W1 * reg

    return grads, loss

  def classify(self, data ):
    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1']
    b2 = self.params['b2']

    scores = np.dot(np.maximum(np.dot(data, W1), 0), W2) + b2
    guesses = np.argmax(scores, axis=1)
    return guesses
  #
  # def sigmoid(self, X):
  #   return 1/(1+np.exp(-X))