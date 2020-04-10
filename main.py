# -*- coding: utf-8 -*-

from models import DeepModels

import os
import numpy as np #arrays e matrizes multidimensionais
import pandas as pd # manipulação e análise de dados

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import find
import seaborn as sn
#%%
#função que plota a acurácia do treinamento e validação. Também plota a função loss do treinamento e validação
def plot_history(history, title, net):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.title('Training and validation accuracy: ' + title)
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()
    filename='trainingvalidationacc'+net
    plt.savefig(filename + '.png')
    
    plt.figure()
    plt.title('Training and validation loss: ' + title)
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.legend()
    filename='trainingvalidationaloss'+net
    plt.savefig(filename + '.png')
    
    plt.show()
    
#função que calcula a acurácia do modelo.
def evaluate_accuracy(model):
    predicted = model.predict(x_val)
    diff = y_val.argmax(axis=-1) - predicted.argmax(axis=-1)
    corrects = np.where(diff == 0)[0].shape[0]
    total = y_val.shape[0]
    return float(corrects/total)
    
#abertura do arquivo da base.
df = pd.read_json('datasets/News_Category_Dataset_v2.json', lines=True)
df.head()

#%% pré-processamento - Juntando algumas classes equivalentes
df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

df.category = df.category.map(lambda x: "ARTS" if x == "ARTS & CULTURE" else x)

df.category = df.category.map(lambda x: "STYLE" if x == "STYLE & BEAUTY" else x)

df.category = df.category.map(lambda x: "PARENTING" if x == "PARENTS" else x)

df.category = df.category.map(lambda x: "HEALTHY LIVING" if x == "WELLNESS" else x)

df.category = df.category.map(lambda x: "ENVIRONMENT" if x == "GREEN" else x)

df.category = df.category.map(lambda x: "FOOD & DRINK" if x == "TASTE" else x)

df.category = df.category.map(lambda x: "EDUCATION" if x == "COLLEGE" else x)

df.category = df.category.map(lambda x: "LATINO VOICES" if x == "GROUPS VOICES" else x)
#%%
#pré-processamento - analisando as linhas que não tem short description e headline

len(df[df['short_description' and 'headline'] == '']) #6 linhas não possuem short description e headline

#%%
#pré-processamento - removendo as linhas que não tem short description e headline

df.drop(df[df['short_description' and 'headline'] == '' ].index, inplace=True)

#%%
#%% juntando o titulo e a breve descrição 
#criando uma nova coluna chamada text
df['text'] = df.headline + " " + df.short_description
#%%
#Tokenização
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text) #cria um vocabulario e atribui um id pra cada palavra
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X
#%%
#pre-processamento no texto
#retira noticias que tem numero de palavras menores que 5
df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]
#%%
#trocando categoria por id
cates = df.groupby('category')
print("total categories:", cates.ngroups)
print(cates.size())

cat_list = cates.size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(cat_list):
    category_int.update({k:i})
    int_category.update({i:k})

df['c2id'] = df['category'].apply(lambda x: category_int[x])
#%%
#padronizando o tamanho da sentence
maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))
#%%
#Algoritmo de pesagem das palavras com referência ao artigo Guo et al. (2019)
tokens = dict()
tokens_inv = dict()
for word, idx in tokenizer.word_index.items():
    tokens[idx] = word
    tokens_inv[word] = idx

vocab_size = len(tokens)
#tdm=matriz termo-documento
tdm = dict()
vectTFIDF = dict()
for k, gp in cates:
    print("Creating TDM for:", k)
    vectTFIDF[k] = TfidfVectorizer(lowercase=True, dtype=np.float32)
    tdm[k] = vectTFIDF[k].fit_transform(gp.text)

weight_vector = dict()
vocab = set(tokens.values())

for k, _ in cates:
    if not os.path.isfile('aux_data/new_weight_vector_'+k+'.npy'):
        print("Creating weight vector for:", k)
        weight_vector[k] = np.zeros(vocab_size+1, dtype='float32')    
        feature_names = set(vectTFIDF[k].get_feature_names())        
        inter = vocab.intersection(feature_names)
        diff = vocab.difference(feature_names)
        min_val = 1
        for word in inter:
            term = vectTFIDF[k].vocabulary_.get(word)           
            weight = max(find(tdm[k][:,term])[2])
            weight_vector[k][tokens_inv[word]] = weight
            if weight < min_val:
                min_val = weight
        for word in diff:
            weight_vector[k][tokens_inv[word]] = min_val 
        np.save('aux_data/new_weight_vector_'+k, weight_vector[k])
    else:
        weight_vector[k] = np.load('aux_data/new_weight_vector_'+k+'.npy')

#%%
#glove embedding
#cada palavra representa um vetor 
EMBEDDING_DIM = 100

embeddings_index = {}
f = open('datasets/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s unique tokens.' % vocab_size)
print('Total %s word vectors.' % len(embeddings_index))
#%%
#Algoritmo de pesagem das palavras com referência ao artigo Guo et al. (2019)
embedding_matrix = np.zeros((vocab_size+1, EMBEDDING_DIM))

for i in range(1, vocab_size+1):
    embedding_vector = embeddings_index.get(tokens[i])
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)

embedding_matrix_final = np.zeros((cates.ngroups,vocab_size+1, EMBEDDING_DIM))        
alist = []
for k, _ in cates:
    x = weight_vector[k]
    x.shape = (len(x), 1)
    alist.append(np.multiply(x, embedding_matrix))
        
embedding_matrix_final=np.stack(alist) #matriz dos I's final. Cada I é uma matriz de pesos correspondente para cada classe
#%%
#Preparação dos dados que irão entrar para a rede neural
X = np.array(X)
Y = np_utils.to_categorical(list(df.c2id))

seed = 29
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
#%%
#Cria uma instância da classe Deep Models
models = DeepModels(maxlen, embedding_matrix_final)
#cria um modelo para a rede attention_lstm
AttentionLSTM = models.create_attention_lstm()
AttentionLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
AttentionLSTM.summary() #imprime os modelos que serão utilizados
#%%Treina o modelo para um número fixo de épocas.
attlstm_history = AttentionLSTM.fit(x_train, 
                                    y_train, 
                                    batch_size=128, 
                                    epochs=10, 
                                    validation_data=(x_val, y_val))
#%%Cria um modelo para a rede bi-gru
BiGRU = models.create_bi_gru_cnn()
BiGRU.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
BiGRU.summary()
#%%Treina o modelo para um número fixo de épocas.
bigru_history = BiGRU.fit(x_train, 
                            y_train, 
                            batch_size=128, 
                            epochs=10, 
                            validation_data=(x_val, y_val))
#%%plota os gráficos de acurácia e loss
plot_history(bigru_history, 'Bidirectional GRU', 'GRU')
plot_history(attlstm_history, 'Attention LSTM', 'LSTM')

#%%imprime a acurácia
print("Attention LSTM accuracy:     %.6f" % evaluate_accuracy(AttentionLSTM))
print("Bidirectional GRU accuracy:  %.6f" % evaluate_accuracy(BiGRU))
#%%
#matriz de confusão
predicted = AttentionLSTM.predict(x_val)
cm = pd.DataFrame(confusion_matrix(y_val.argmax(axis=1), predicted.argmax(axis=1)))
#%%
#heatmap para o LSTM
plt.figure(figsize=(15, 10))
ax = sn.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
ax.set_ylabel('Predicted')
ax.set_xlabel('Target');
ax.set_title("Matriz de Confusão para o conjunto de teste", size=12)
plt.savefig('matrizconfusaolstm.jpg')
#%%matriz de confusão
predicted = BiGRU.predict(x_val)
cm = pd.DataFrame(confusion_matrix(y_val.argmax(axis=1), predicted.argmax(axis=1)))
#%%heatmap para o GRU
plt.figure(figsize=(15, 10))
ax = sn.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
ax.set_ylabel('Predicted')
ax.set_xlabel('Target');
ax.set_title("Matriz de Confusão para o conjunto de teste", size=12)
plt.savefig('matrizconfusaoGRU.jpg')
