# Classificação de notícias utilizando ponderação de embeddings e redes GRU Bidirecional e Attention LSTM.

O algoritmo desenvolvido foi baseado no artigo de Guo et al. (2019) [1], onde obtêm-se uma matriz termo-documento utilizando uma pesagem pela equação TF-IDF. Para a classificação do texto News_Category_Dataset.json (página: https://www.kaggle.com/rmisra/news-category-dataset) foram utilizadas duas redes neurais: GRU Bidirecional e Attention LSTM.

Para rodar o código siga as seguintes instruções:

Dentro da pasta datasets deverá conter 2 arquivos. 

O primeiro arquivo seria o News_Category_Dataset_v2.json. Para baixar o arquivo entre em: https://www.kaggle.com/rmisra/news-category-dataset, baixe o arquivo news-category-dataset.zip clicando no link download. Após isso, é só descompactar o arquivo e colocá-lo dentro da pasta datasets.

O segundo arquivo seria o glove.6B.100d.txt. Para baixar o arquivo entre em: https://nlp.stanford.edu/projects/glove/, procure por glove.6B.zip, descompacte e coloque o arquivo glove.6B.100d.txt dentro da pasta datasets.

Dentro da pasta aux_data já contém os arquivos de pesos para cada classe. Se preferir que o programa gere novamente é só deletar todos os arquivos e deixar a pasta vazia.

Para rodar o código, digite python3 main.py no terminal do Ubuntu.

O programa gera como resultado as curvas de Acurácia e loss do treinamento e validação. Além disso, o valor acurácia e a matriz de confusão para ambas as redes também são mostradas.


Referências Bibliográficas:

[1]  Bao  Guo,  Chunxia  Zhang,  Junmin  Liu,  and  Xiaoyi  Ma.   Improving  text  clas-sification  with  weighted  word  embeddings  via  a  multi-channel  textcnn  model.Neurocomputing, 363:366 – 374, 2019.


