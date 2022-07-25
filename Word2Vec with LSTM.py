# Word Embedding + LSTM

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Flatten
from tensorflow.keras import optimizers

vocabulary_size = 5000 # 5000 unique words in the corpus
max_words = 500 # only use 500 words from each review
embedding_size=32 # use a 32-dimension vector to represent a word
epoches = 50
state_dim = 32

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))


word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, ' ') for i in X_train[6]])
print('---label---')
print(y_train[6])

print('Maximum review length: {}'.format(len(max((X_train + X_test), key=len))))
print('Minimum review length: {}'.format(len(min((X_test + X_test), key=len))))

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


model=tf.keras.models.Sequential()

model.add(Embedding(vocabulary_size,embedding_size,input_length=max_words))
# totally 5000 words and every word vector has 32 components; and 1 review has 500 words
# 5000 x 32 = 160,000 parameters

#model.add(Flatten())
# input 500 words, and every word vector has 32 components;
# 500 x 32 = 16,000 input values

model.add(LSTM(state_dim,return_sequences=False))
model.add(Dense(1,activation='sigmoid')) # Logistic Regression, 16000 w + 1 b, totally 16001 parameters

print(model.summary())

X_test, y_test = X_test[:2000], y_test[:2000]

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001), metrics=['accuracy'])


history = model.fit(X_train,y_train,epochs=epoches,batch_size=64,validation_data=(X_test,y_test))


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()