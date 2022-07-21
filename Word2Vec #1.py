import numpy as np
import tensorflow as tf

class Word2Vec:
  def __init__(self, vocab_size=0, embedding_dim=16, optimizer='sgd', epochs=10000):
    self.vocab_size=vocab_size
    self.embedding_dim=5
    self.epochs=epochs
    if optimizer=='adam':
      self.optimizer = tf.optimizers.Adam()
    else:
      self.optimizer = tf.optimizers.SGD(learning_rate=0.1)

  def train(self, x_train=None, y_train=None):
    self.W1 = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_dim])) # 5x7 dimension
    self.b1 = tf.Variable(tf.random.normal([self.embedding_dim])) # 5x1 dimension
    self.W2 = tf.Variable(tf.random.normal([self.embedding_dim, self.vocab_size])) # 7x5 dimension
    self.b2 = tf.Variable(tf.random.normal([self.vocab_size])) # 7x1 dimension
    for _ in range(self.epochs):
      with tf.GradientTape() as t:
        # there is only one hidden layer in this code
        hidden_layer = tf.add(tf.matmul(x_train,self.W1),self.b1) # 34x7 * 7x5 + 5 scalars = 34x5
        # the output layer uses softmax regression to finalize the output
        output_layer = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, self.W2), self.b2)) # 34x5 * 5x7 + 7 scalars = 34x7
        # cross entropy is used as the loss function because all the output values are between 0 and 1
        cross_entropy_loss = tf.reduce_mean(-tf.math.reduce_sum( y_train * tf.math.log(output_layer), axis=[1]))
      # performing optimization on each variable
      grads = t.gradient(cross_entropy_loss, [self.W1, self.b1, self.W2, self.b2])
      t1 = zip(grads,[self.W1, self.b1, self.W2, self.b2])
      self.optimizer.apply_gradients(t1)
      if(_ % 1000 == 0):
        print(cross_entropy_loss)

  def vectorized(self,index):
    return (self.W1+self.b1)[index]

initalPhrase = 'He is the king . The king is royal . She is the royal  queen '
rawSentences = initalPhrase.lower().split('.')

sentences = []
for sentence in rawSentences:
    sentences.append(sentence.split())

# create an list that contains every combination of each word with its two closest neighbors on each side
# The first and last word only have two neighbors each while the second and second last word only have three neighbors each
data = []
windowSize = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for neighborWord in sentence[max(word_index - windowSize, 0):min(word_index + windowSize, len(sentence)) + 1]:
            if neighborWord != word:
                data.append([word, neighborWord])

# creating a list that contains every distinctive word in the initial phrase
distinctiveWords = []
for word in initalPhrase.split():
    if word != '.':
        distinctiveWords.append(word)
distinctiveWords = set(distinctiveWords)
vocab_size = len(distinctiveWords)

# creating two directories using all indexes and words
word2int = {}
int2word = {}
for i,word in enumerate(distinctiveWords):
    word2int[word] = i
    int2word[i] = word
    
# performing one-hot encoding on one word
def to_one_hot(index, length):
    oneHotArray = np.zeros(length)
    oneHotArray[index] = 1
    return oneHotArray

x_train = [] # input word
y_train = [] # output word
for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))
x_train = np.asarray(x_train, dtype='float32')
y_train = np.asarray(y_train, dtype='float32')
w2v = Word2Vec(vocab_size=vocab_size, optimizer='adam', epochs=10000)
w2v.train(x_train, y_train)

print(w2v.W1 + w2v.b1)

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

print(int2word[find_closest(word2int['king'], vectors)])
print(int2word[find_closest(word2int['queen'], vectors)])
print(int2word[find_closest(word2int['royal'], vectors)])

from sklearn.manifold import TSNE
from sklearn import preprocessing
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)
normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlim(left=-1, right=1)
ax.set_ylim(bottom=-1, top=1)
for word in words:
    print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
plt.show()
