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
    self.W1 = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_dim]))
    self.b1 = tf.Variable(tf.random.normal([self.embedding_dim])) #bias
    # W1, 5x7; b1, 5x1; X, 7x1
    # 5x7 x 7x1 + 5x1 = 5x1

    self.W2 = tf.Variable(tf.random.normal([self.embedding_dim, self.vocab_size]))
    self.b2 = tf.Variable(tf.random.normal([self.vocab_size]))
    # W2, 7x5; b2, 7x1; X, 5x1
    # 7x5 x 5x1 + 7x1 = 7x1

    for _ in range(self.epochs):
      with tf.GradientTape() as t:
        hidden_layer = tf.add(tf.matmul(x_train,self.W1),self.b1)
        output_layer = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, self.W2), self.b2))
      #  cross_entropy_loss = tf.reduce_mean( -tf.math.reduce_sum( y_train * tf.math.log(output_layer), axis=[1] ) )
        temp1 = tf.math.log(output_layer)
        temp2 = -tf.math.reduce_sum(y_train *temp1, axis=[1])
        cross_entropy_loss = tf.reduce_mean(temp2)
      grads = t.gradient(cross_entropy_loss, [self.W1, self.b1, self.W2, self.b2])
      t1 = zip(grads,[self.W1, self.b1, self.W2, self.b2])
      self.optimizer.apply_gradients(t1)

      if(_ % 1000 == 0):
        print(cross_entropy_loss)

  def vectorized(self, word_idx):
    return (self.W1+self.b1)[word_idx]

corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
corpus_raw = corpus_raw.lower()
# raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

data = []
WINDOW_SIZE = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):

        # for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
        temp1 = max(word_index - WINDOW_SIZE, 0)
        temp2 = min(word_index + WINDOW_SIZE, len(sentence)) + 1
        for nb_word in sentence[temp1:temp2 ]:
            if nb_word != word:
                data.append([word, nb_word])

words = []
for word in corpus_raw.split():
    if word != '.': # because we don't want to treat . as a word
        words.append(word)
words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word


# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append( to_one_hot(word2int[ data_word[0] ], vocab_size) )
    y_train.append( to_one_hot(word2int[ data_word[1] ], vocab_size) )
# split the training data to the input words and output words
# use one-hot vectors to represent the words

# convert them to numpy arrays
x_train = np.asarray(x_train, dtype='float32')
y_train = np.asarray(y_train, dtype='float32')

# instantiation, same thing as creating an object with parameters
w2v = Word2Vec(vocab_size=vocab_size, optimizer='adam', epochs=10000)
w2v.train(x_train, y_train)


temp = w2v.vectorized(word2int['queen'])
print(temp)


vectors = w2v.W1 + w2v.b1
print(vectors)

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


print("end")