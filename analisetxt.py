import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split


def load_imdb_dataset(max_features, maxlen):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train, y_train), (x_test, y_test)


def build_word_index():
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    return word_index


def build_index_to_word(word_index):
    index_to_word = {v: k for k, v in word_index.items()}
    return index_to_word


def build_model(max_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_features, 64),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, learning_rate):
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    return history


def save_model_weights(model, filename):
    model.save_weights(filename)


def save_model_architecture(model, filename):
    with open(filename, 'w') as json_file:
        json_file.write(model.to_json())


def save_word_index(word_index, filename):
    with open(filename, 'w') as json_file:
        json.dump(word_index, json_file)


def save_index_to_word(index_to_word, filename):
    with open(filename, 'w') as json_file:
        json.dump(index_to_word, json_file)


def plot_metrics(history, epochs):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

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


# Parâmetros
max_features = 25000
maxlen = 300
batch_size = 64
epochs = 3
learning_rate = 0.001

# Carregar o dataset IMDB
(x_train, y_train), (x_test, y_test) = load_imdb_dataset(max_features, maxlen)

# Divisão em conjunto de treinamento e conjunto de teste
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Construir os índices de palavras
word_index = build_word_index()
index_to_word = build_index_to_word(word_index)

# Construir o modelo
model = build_model(max_features)

# Treinar o modelo
history = train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, learning_rate)

# Salvar os pesos do modelo
save_model_weights(model, 'modelo_pesos.h5')

# Salvar a arquitetura do modelo em formato JSON
save_model_architecture(model, 'modelo_arquitetura.json')

# Salvar o índice de palavras
save_word_index(word_index, 'word_index.json')

# Salvar o índice inverso de palavras
save_index_to_word(index_to_word, 'index_to_word.json')

# Plotar o gráfico de métricas
plot_metrics(history, epochs)

# Avaliar o desempenho do modelo no conjunto de teste
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia: {accuracy * 100:.2f}%")
print(f"Perda: {loss * 100:.2f}%")
