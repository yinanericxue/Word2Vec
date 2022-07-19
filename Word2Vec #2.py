import collections
import os
import random
import urllib
import zipfile
import numpy as np
import tensorflow as tf

learningRate = 0.1
batchSize = 128
epochs = 300000
displayStep = 10000
evaluationStep = 200000

# all words that are being analyzed
evaluationWords = ['nine', 'of', 'going', 'hardware', 'american', 'britain']

# eachh word vector dimension after embedding
embeddingSize = 200

# the maximum amount of words that are stored
maxDictionarySize = 50000

# the minimum amount of times a word must appear in the corpus for it to be stored in the dictionary
minFrequency = 10

# three words on each side are chosen during Skipgram
windowSize = 3

# two words are picked out of the six words chosen during Skipgram
valueSelectionAmount = 2 # randomly select 2 from the window

# 64 negative samples are added to every 128 actual samples
negativeSamplingAmount = 64

data_path = 'text8.zip'

# 17,005,207 words in the corpus
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

# 'UNK' will represent all the words that appeared less than 10 times in the corpus
count = [('UNK', -1)]

# all the 49999 words that appeared in the corpus are added to "count", arranged in the tuple form ("word","frequency amount")
count.extend( collections.Counter(text_words).most_common(maxDictionarySize - 1) )

# "count" only keeps the words that appeared at least 10 times in the corpus, 47134 words + UNK remaining
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < minFrequency:
        count.pop(i)
    else:
        break
    
# 47135 total, 47134 + UNK
vocabulary_size = len(count)

word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:   #  traverse the corpus, get the ID of each word and save it into the 'data'
    index = word2id.get(word, 0) # Get the ID of each word
                                 # Return 0 if the key(word) doesn't exist in the dict
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count) # totally 444176 words in the corpus are not in the vocabulary and replaced with "UNK"
id2word = dict( zip(word2id.values(), word2id.keys()) )

data_index = 0

def next_batch(batchSize,valueSelectionAmount,windowSize):
    global data_index
    assert batchSize % valueSelectionAmount == 0
    assert valueSelectionAmount <= 2 * windowSize
    batch = np.ndarray(shape=(batchSize), dtype=np.int32)
    labels = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
    span = 2 * windowSize + 1 # 7
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data): # reach to the end of corpus, go to the beginning
        data_index = 0
    buffer.extend( data[data_index:data_index + span] )
    data_index += span
    temp = batchSize // valueSelectionAmount
    for i in range(temp):
        context_words = [w for w in range(span) if w != windowSize] # [0, 1, 2, 4, 5, 6], 3 is input
        words_to_use = random.sample(context_words,valueSelectionAmount) # select 2 words as output
        for j, context_word in enumerate(words_to_use):
            batch[i * valueSelectionAmount + j] = buffer[windowSize] # 3 is input word
            labels[i * valueSelectionAmount + j, 0] = buffer[context_word] # output word
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index]) # slide the window by 1
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

with tf.device('/cpu:0'):
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embeddingSize])) #维度：47135, 200
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embeddingSize]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

def get_embedding(x):
    with tf.device('/cpu:0'):
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed

def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss( weights=nce_weights,biases=nce_biases,labels=y,inputs=x_embed,
                            num_sampled=negativeSamplingAmount, # how many negative samples generated
                            num_classes=vocabulary_size)
                           )
        return loss

def evaluate(x_embed):
    with tf.device('/cpu:0'):
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))#归一化
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32)#全部向量的
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)#计算余弦相似度
        return cosine_sim_op

optimizer = tf.optimizers.SGD(learningRate)

def run_optimization(x, y):
    with tf.device('/cpu:0'):
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = nce_loss(emb, y)
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])
        optimizer.apply_gradients( zip(gradients, [embedding, nce_weights, nce_biases]) )
        
x_test = np.array([word2id[w.encode('utf-8')] for w in evaluationWords])

for step in range(1, epochs + 1):
    batch_x, batch_y = next_batch(batchSize,valueSelectionAmount,windowSize)
    run_optimization(batch_x, batch_y)
    if step % displayStep == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))
    if step % evaluationStep == 0 or step == 1:
        print("Evaluation...")
        sim = evaluate(get_embedding(x_test)).numpy()
        for i in range(len(evaluationWords)):
            top_k = 8  # 8 most similar words
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % evaluationWords[i]
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, id2word[nearest[k]])
            print(log_str)