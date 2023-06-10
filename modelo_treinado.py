import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import sequence

import json

maxlen = 30

# Carregar a arquitetura do modelo
with open('modelo_arquitetura.json', 'r') as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)

# Carregar os pesos do modelo
loaded_model.load_weights('modelo_pesos.h5')

# Carregar o índice de palavras
with open('word_index.json', 'r') as json_file:
    word_index = json.load(json_file)

# Carregar o índice inverso de palavras
with open('index_to_word.json', 'r') as json_file:
    index_to_word = json.load(json_file)

while True:
    # Solicita entrada do usuário
    text = input("Digite o texto (ou 'sair' para encerrar): ")

    # Verifica se o usuário deseja sair
    if text.lower() == 'sair':
        break

    # Pré-processa o texto
    text_vector = sequence.pad_sequences([[word_index[word] for word in text.lower().split()]], maxlen=maxlen)

    # Faz a previsão
    prediction = loaded_model.predict(text_vector)[0][0]

    # Interpreta a previsão
    if prediction >= 0.5:
        print("O sentimento associado ao texto é positivo.")
    else:
        print("O sentimento associado ao texto é negativo.")

