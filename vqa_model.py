from keras import utils
from keras import layers
from keras import models
from keras import applications


def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
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


def img_model(dropout_rate, shape=(64, 64, 3)):
    print("Creating image model...")
    inputs = layers.Input(shape=shape)
    model = applications.VGG16(input_tensor=inputs, weights='imagenet', include_top=False)
    model.add(layers.Dense(1024, input_dim=4096, activation='tanh'))
    return model


def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model(dropout_rate)
    text_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")
    fc_model = models.Sequential()
    fc_model.add(layers.multiply([vgg_model, text_model]))
    fc_model.add(layers.Dropout(dropout_rate))
    fc_model.add(layers.Dense(1000, activation='tanh'))
    fc_model.add(layers.Dropout(dropout_rate))
    fc_model.add(layers.Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model
