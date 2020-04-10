# -*- coding: utf-8 -*-

from attention import Attention

from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, GRU
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Concatenate
from keras.models import Model
from keras.initializers import Constant

#%%Classes construídas para os modelos GRU bidimensional e Attention LSTM

class DeepModels():
    def __init__(self, maxlen, embeddings):
        self.embeddings = embeddings
        self.ngroups = embeddings.shape[0]
        self.vocab_size = embeddings.shape[1]
        self.embedding_dim = embeddings.shape[2]
        self.maxlen = maxlen
        
    def create_attention_lstm(self):        
        lstm_layer = LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)
        inp = Input(shape=(self.maxlen,), dtype='int32')
        class_layers = []
        for i in range(self.ngroups):
            embedding = Embedding(self.vocab_size,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embeddings[i]),
                                    input_length=self.maxlen,
                                    trainable=False)(inp)
            class_layers.append(embedding)
           
        merged = Concatenate()(class_layers)
        x = lstm_layer(merged)
        x = Dropout(0.25)(x)
        x = Attention(self.maxlen)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        outp = Dense(self.ngroups, activation='softmax')(x)
        
        return Model(inputs=inp, outputs=outp)
    
    def create_bi_gru_cnn(self):
        inp = Input(shape=(self.maxlen,), dtype='int32')
        class_layers = []
        for i in range(self.ngroups):
            embedding = Embedding(self.vocab_size,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embeddings[i]),
                                    input_length=self.maxlen,
                                    trainable=False)(inp)
            class_layers.append(embedding)
            
        merged = Concatenate()(class_layers)
        x = SpatialDropout1D(0.2)(merged)
        x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
        x = Conv1D(64, kernel_size=3)(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, max_pool])
        outp = Dense(self.ngroups, activation="softmax")(x)
        
        return Model(inp, outp)