from keras import utils
from keras import layers
from keras import models
from keras import applications
from dsnet import dense_net


def text_model(inputs, num_words=1311, embedding_dim=256, seq_length=19):
    print("Creating text model...")

    embedded_question = layers.Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=seq_length)(inputs)
    encoded_question = layers.LSTM(256)(embedded_question)
    question_encoder = models.Model(inputs=inputs, outputs=encoded_question)
    return question_encoder


def visual_model(inputs):
    print("Creating image model...")
    model = dense_net([6, 12, 24, 16])
    encoded_frame = layers.TimeDistributed(model)(inputs)
    encoded_video1 = layers.LSTM(1024)(encoded_frame)
    encoded_video2 = layers.Dropout(0.5)(encoded_video1)
    encoded_video3 = layers.Dense(512, activation="relu")(encoded_video2)
    encoded_video4 = layers.Dropout(0.2)(encoded_video3)
    encoded_video = layers.Dense(256, activation="relu")(encoded_video4)
    return encoded_video


def vqa_model(shape=(10, 64, 64, 3)):
    video_input = layers.Input(shape=shape)
    question_input = layers.Input(shape=(16,), dtype='int32')
    merged = layers.concatenate([visual_model(video_input), text_model(question_input)])
    output = [layers.Dense(1311, activation='softmax')(merged)]
    video_qa_model = models.Model(inputs=[video_input, question_input], outputs=output)
    video_qa_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    video_qa_model.summary()
    return video_qa_model

