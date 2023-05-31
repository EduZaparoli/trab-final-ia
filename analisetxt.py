import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# Carrega o dataset IMDB
max_features = 80000
maxlen = 30
batch_size = 128
epochs = 17
learning_rate = 0.001
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Obtém o índice de palavras do dataset
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2

# Inverte o índice de palavras
index_to_word = {v: k for k, v in word_index.items()}

# Define o modelo de rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features, 64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(128), # aumentado de 64 para 128
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy'])

# Treina o modelo
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Capturar as métricas de precisão durante o treinamento
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Salvar os pesos do modelo
model.save_weights('modelo_pesos.h5')

# Salvar a arquitetura do modelo em formato JSON
with open('modelo_arquitetura.json', 'w') as json_file:
    json_file.write(model.to_json())

# Salvar o índice de palavras
import json

with open('word_index.json', 'w') as json_file:
    json.dump(word_index, json_file)

# Salvar o índice inverso de palavras
with open('index_to_word.json', 'w') as json_file:
    json.dump(index_to_word, json_file)

# Plotar o gráfico de precisão
epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, train_loss, label='Loss (Training)')
plt.plot(epochs_range, val_loss, label='Loss (Validation)')
plt.plot(epochs_range, train_accuracy, label='Accuracy (Training)')
plt.plot(epochs_range, val_accuracy, label='Accuracy (Validation)')
plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.show()
