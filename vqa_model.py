from keras import utils
from keras import layers
from keras import models
from keras import applications
from dsnet import dense_net


def text_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating text model...")
    model = models.Sequential()
    model.add(layers.Embedding(num_words, embedding_dim,
                               weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(layers.LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.LSTM(units=512, return_sequences=False))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1024, activation='tanh'))
    return model


def img_model(shape=(64, 64, 3)):
    print("Creating image model...")
    inputs = layers.Input(shape=shape)
    layers.TimeDistributed()
    model = dense_net(inputs,[6,12,24,16])



def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    pass

img_model()