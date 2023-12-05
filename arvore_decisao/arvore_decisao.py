import os
import pandas as pd
import pydotplus
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

#carregando o dataset
df = pd.read_csv("Hotel Reservations.csv")

#imprimindo infomações do dataframe para análise
print(df.shape)
print(df.info())
print(df.head(20)) 

#imprimindo dados vazios ou duplicados
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe().T)

#explorando valores unicos de colunas do tipo objeto (categorias pré-definidas) 
print(df['type_of_meal_plan'].unique())
print(df['room_type_reserved'].unique())
print(df['market_segment_type'].unique())
print(df['booking_status'].unique())

#realizando o one hot encoding e substituindo o dataframe original pelo codificado
df_codificado = pd.get_dummies(df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status'], drop_first = True)

df_codificado.to_csv("arvore_decisao/" + "Hotel_Reservations_Codificado.csv", index=False)

print(df_codificado.info())

print('---------------------------------------------------------------------------------------------')

#------------------ IMPLEMENTANDO A ÁRVORE DE DECISAO SEM CROSS VALIDATION E GRID SEARCH ------------------
#removendo colunas que não serão utilizadas em X e definindo y
X = df_codificado.drop(columns=['Booking_ID', 'booking_status_Not_Canceled'])
y = df_codificado['booking_status_Not_Canceled']

#realizando a divisão em dados de treinamento e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#criando a árvore
clf = DecisionTreeClassifier(random_state=42)

#acurácia e f1 do modelo sem otimização
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acuracia_normal = accuracy_score(y_test, y_pred)
f1score_normal = f1_score(y_test, y_pred)
print('Acurácia do modelo sem otimização no conjunto de teste: ', acuracia_normal)
print('F1 score do modelo sem otimização no conjunto de teste: ', f1score_normal)
print('---------------------------------------------------------------------------------------------')

#------------------IMPLEMENTANDO GRID SEARCH E CROSS VALIDATION NA ÁRVORE DE DECISÃO ------------------
#definindo hiperparâmetros para otimizar o modelo
hiperparametros = {'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
                   'criterion': ['gini', 'entropy']}

grid_search = GridSearchCV(clf, hiperparametros, cv=5, scoring='accuracy')

# treinando o modelo e obtendo a melhor combinação de hiperparâmetros
grid_search.fit(X_train, y_train)
melhor_hiperparametro = grid_search.best_params_
print("Melhores hiperparâmetros:", melhor_hiperparametro)

#obtendo o melhor modelo encontrado durante o grid search
best_model = grid_search.best_estimator_

#fazendo previsões no conjunto de teste e calculando a acurácia
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)
acuracia = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
acuracia_treino = accuracy_score(y_train, y_pred_train)
f1score_treino = f1_score(y_train, y_pred_train)
print('Acurácia do modelo com grid search e cross validation no conjunto de teste: ', acuracia)
print('F1 score do modelo com grid search e cross validation no conjunto de teste: ', f1score)

#------------------ GRÁFICOS E RELATÓRIOS ------------------

#gráfico de importância das características 
feature_importances = best_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Importância das características')
plt.ylabel('Características')
plt.title('Importância das características na Árvore de Decisão')
# plt.show()
plt.savefig(os.path.join("arvore_decisao/", 'importancia_caracteristicas.png'))

#gerando a matriz de confusão
confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de confusão para o conjunto de teste na árvore de decisão')
plt.savefig(os.path.join("arvore_decisao/", 'matriz_confusao.png'))

#gerando o relatório de classificação
relatorio_classificacao = classification_report(y_test, y_pred)
print("Relatório de classificação:\n", relatorio_classificacao)

#gerando a árvore de decisão
dot_data = export_graphviz(best_model, out_file=None, feature_names=X.columns, class_names=['Not Canceled', 'Canceled'], filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('arvore_decisao.png')
graph.write_png(os.path.join("arvore_decisao/", 'arvore_decisao.png'))