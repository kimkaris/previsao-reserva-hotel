# Prevendo a chance de clientes de hotéis cancelarem ou manterem reservas realizadas
A possibilidade de realizar reservas online em hotéis mudou drasticamente o comportamento de clientes. Os hotéis enfrentam grandes problemas devido ao cancelamento de reservas ou o não comparecimento dos hóspedes, devido a muitos hotéis não terem taxa de cancelamento ou esta ser muito baixa, fazendo com que o hotel saia no prejuízo.

### Objetivo
- Utilizar a seguinte base de dados do Kaggle para desenvolver o trabalho: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset;
- Escolher uma resolução desse problema realizada por algum usuário, com a finalidade de tentar obter um resultado melhor (escolhemos esta, que utiliza Floresta Aleatória e a chamaremos de técnica A: https://www.kaggle.com/code/battle11king/hotel-reservation);
- Utilizar técnicas de aprendizagem de máquina para prever qual método é melhor para descobrir se um cliente seguirá com sua reserva em um hotel ou irá cancelar;
- Comparar os resultados das técnicas a fim de obter a com melhor desempenho;
- Verificar se é possível obter um melhor resultado utilizando  a técnica de *cross-validation* com *grid search*

### Contexto dos dados
- Nossa base de dados possui informações de reservas de clientes em um hotel;
- São 19 colunas colunas na tabela, contendo informações como ID, dia, mês e ano da reserva, quantidade de adultos, quantidade de crianças, quantidade de dias que possuía a reserva, preço médio por quarto e se ela foi cancelada ou não; 
- A base não possui dados duplicados e não possui colunas sem valores. Ou seja, não foi necessário realizar ajustes para tratar essas exceções.

### Técnicas
- Floresta aleatória (técnica A):
  - Floresta aleatória é um algoritmo de machine learning comumente usado que combina a saída de várias árvores de decisão para alcançar um único resultado. Sua facilidade de uso e flexibilidade incentivaram a sua adoção, pois lida tanto com problemas de classificação quanto de regressão;
  - Os algoritmos de floresta aleatória possuem três hiperparâmetros principais, os quais devem ser definidos antes do treinamento. Estes incluem o tamanho do nó, o número de árvores e o número de recursos amostrados.
- Árvore de decisão (técnica B):
  - Árvores de decisão é um dos métodos mais utilizados para a tomada de decisão em modelos de aprendizagem de máquina. Também é usado em pesquisa operacional;
  - É um modelo que representa um conjunto de decisões e resultados possíveis em uma estrutura hierárquica de nós e ramos, onde cada nó representa um teste em um atributo e os ramos representam os resultados possíveis desse teste;
- K-Nearest Neighbors (técnica C)
  - KNN é um algoritmo classificador de aprendizagem supervisionada. O algoritmo calcula a distância entre a instância a ser classificada e todas as outras instâncias no conjunto de treinamento;
  - Identifica os k vizinhos mais próximos (mais semelhantes) com base na medida de distância, geralmente usando a distância euclidiana;
  - Para classificação, o KNN atribui à instância a classe que é mais comum entre seus k vizinhos mais próximos. Em outras palavras, a classe mais frequente entre os vizinhos determina a classificação da instância. 
- Gradient Boosting (técnica D)
  - É uma técnica Boosting de aprendizado de máquina que produz modelos de previsão com base em um conjunto de modelos de previsão fracos – geralmente, árvores de decisão;
  - O objetivo do algoritmo é criar uma corrente de modelos fracos, onde cada um tem como objetivo minimizar o erro do modelo anterior, por meio de uma função de perda;
  - Conforme ocorrem os ajustes, o modelo fraco é multiplicado um valor: a taxa de aprendizagem. Ele tem como objetivo determinar o impacto de cada árvore no modelo final;
  - Quanto menor esse valor, menor a contribuição de cada árvore.

### Resultados
![image](https://github.com/user-attachments/assets/55ec0dfa-20da-4fba-b370-4e00a4f2958f)
![image](https://github.com/user-attachments/assets/7cd783a4-d565-4c8e-bf23-2de16474e59e)
![image](https://github.com/user-attachments/assets/f21da8b5-dd82-48c4-abfc-f731c5443fe8)



Após a execução dos códigos e obtenção dos resultados, foi possível constatar que:
- A inserção de cross-validation de fato melhorou os resultados dos nossos experimentos;
- A partir da comparação das três técnicas executadas, foi possível observar que a **árvore de decisão (técnica B)** foi a mais eficiente em questão do resultado de acurácia e de F1;
![image](https://github.com/user-attachments/assets/bf7ae361-e763-4500-969a-27f84656a1b7)
- Não conseguimos superar os resultados da técnica A, porém obtivemos valores próximos e satisfatórios;
- Acreditamos que obtivemos valores altos em acurácia e F1 por se tratarem de dados textuais, sem linhas duplicadas e células nulas.
