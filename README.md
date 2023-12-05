# Prevendo a chance de clientes de hotéis cancelarem ou manterem reservas realizadas
Trabalho realizado para a disciplina de Aprendizagem de Máquina.

## Especificação
A possibilidade de realizar reservas online em hotéis mudou drasticamente o comportamento de clientes. Os hotéis enfrentam grandes problemas devido ao cancelamento de reservas ou o não comparecimento dos hóspedes, devido a muitos hotéis não terem taxa de cancelamento ou esta ser muito baixa, fazendo com que o hotel saia no prejuízo.

### Objetivo
- Utilizar a seguinte base de dados do Kaggle para desenvolver o trabalho: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset;
- Escolher uma resolução desse problema realizada por algum usuário, com a finalidade de tentar obter um resultado melhor (escolhemos esta, que utiliza Floresta Aleatória e a chamaremos de técnica A: https://www.kaggle.com/code/battle11king/hotel-reservation);
- Utilizar técnicas de aprendizagem de máquina para prever se um cliente seguirá com sua reserva em um hotel ou irá cancelar;
- Comparar os resultados das técnicas a fim de obter a com melhor desempenho;
- Verificar se é possível obter um melhor resultado utilizando *cross-validation* com *grid search*

### Técnicas escolhidas
- Árvore de decisão (técnica B)
- K-Nearest Neighbors (técnica C)
- Gradient Boosting (técnica D)

### Resultados
- A inserção de cross-validation de fato melhorou os resultados dos nossos experimentos;
- A partir da comparação das três técnicas executadas, foi possível observar que a árvore de decisão (técnica B) foi a mais eficiente em questão do resultado de acurácia e de F1;
- Não conseguimos superar os resultados da técnica A, porém obtivemos valores próximos e satisfatórios;
- Acreditamos que obtivemos valores altos em acurácia e F1 por se tratarem de dados textuais, sem linhas duplicadas e células nulas 
