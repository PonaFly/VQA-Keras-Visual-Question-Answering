from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Concatenate, Input
from keras.models import  Model
import h5py

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating text model...")
    w2v_input = Input((seq_length,))
    w2v_embed = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=seq_length,
                          weights=[embedding_matrix],trainable=False)(w2v_input)
    w2v_lstm1 = LSTM(512, input_shape=(seq_length, embedding_dim),return_sequences=True)(w2v_embed)
    w2v_drop1 = Dropout(dropout_rate)(w2v_lstm1)
    w2v_lstm2 = LSTM(512, return_sequences=False)(w2v_drop1)
    w2v_drop2 = Dropout(dropout_rate)(w2v_lstm2)
    w2v_dense = Dense(1024, activation='tanh')(w2v_drop2)
    model = Model(w2v_input, w2v_dense)
    return model

def img_model(dropout_rate):
    print("Creating image model...")
    img_input = Input((4096,))
    img_dense = Dense(1024, activation='tanh')(img_input)
    model = Model(img_input, img_dense)
    return model

def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model(dropout_rate)
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")
    fc_concat = Concatenate()([vgg_model.output, lstm_model.output])
    fc_drop1 = Dropout(dropout_rate)(fc_concat)
    fc_dense1 = Dense(1000, activation='tanh')(fc_drop1)
    fc_drop2 = Dropout(dropout_rate)(fc_dense1)
    fc_dense2 = Dense(num_classes, activation='softmax')(fc_drop2)
    fc_model = Model([vgg_model.input, lstm_model.input], fc_dense2)
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
        metrics=['accuracy'])
    return fc_model

