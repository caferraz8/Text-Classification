# Classificação de texto utilizando ponderação de pesos e as redes GRU Bidirecional e Attention LSTM.

Dentro da pasta datasets deverá conter 2 arquivos. 

O primeiro arquivo seria o News_Category_Dataset_v2.json. Para baixar o arquivo entre em: https://www.kaggle.com/rmisra/news-category-dataset, baixe o arquivo news-category-dataset.zip clicando no link download. Após isso, é só descompactar o arquivo e colocá-lo dentro da pasta datasets.

O segundo arquivo seria o glove.6B.100d.txt. Para baixar o arquivo entre em: https://nlp.stanford.edu/projects/glove/, procure por glove.6B.zip, descompacte e coloque o arquivo glove.6B.100d.txt dentro da pasta datasets.

Dentro da pasta aux_data já contém os arquivos de pesos para cada classe. Se preferir que o programa gere novamente é só deletar todos os arquivos e deixar a pasta vazia.

Para rodar o código, digite python3 main.py no terminal do Ubuntu.

O programa gera como resultado as curvas de Acurácia e loss do treinamento e validação. Além disso, o valor acurácia e a matriz de confusão para ambas as redes também são mostradas.


