from keras.layers import  Dense, Embedding, LSTM
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from data_helpers import load_data
import matplotlib.pyplot as plt

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)


sequence_length = x.shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765
embedding_dim = 256
filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5

epochs = 5
batch_size = 80

# Creating the Model
print("Creating Model...")


model = Sequential()
model.add(Embedding(20000, 100, input_length=56))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


## Fit the model

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)


# Evaluate the model

score, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

print('LSTM Score: %.2f' %(score))
print('LSTM Validation Accuracy: %.2f' % (accuracy))


# Plot The LSTM LOSS
fig1 = plt.figure()
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves :LSTM', fontsize=16)
fig1.savefig('loss_lstm.png')


# Plot the LSTM Accuracy

fig2=plt.figure()
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves : LSTM', fontsize=16)
fig2.savefig('accuracy_lstm.png')
plt.show()

