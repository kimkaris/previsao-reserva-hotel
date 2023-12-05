import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r'Hotel Reservations.csv'
reserva = pd.read_csv(file_path)

#tratando dados por label encoding
label_encoder = LabelEncoder()
reserva['type_of_meal_plan'] = label_encoder.fit_transform(reserva['type_of_meal_plan'])
reserva['room_type_reserved'] = label_encoder.fit_transform(reserva['room_type_reserved'])
reserva['market_segment_type'] = label_encoder.fit_transform(reserva['market_segment_type'])
reserva['booking_status'] = label_encoder.fit_transform(reserva['booking_status'])

#remoção de colunas irrelevantes
colunas_para_remover = ['Booking_ID']
reserva = reserva.drop(colunas_para_remover, axis=1)

#divisão de dados
X = reserva.drop('booking_status', axis=1)
y = reserva['booking_status']

print(reserva.info())

#scaler para a normalização dos dados
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

#dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

#inicio do modelo
knn_model = KNeighborsClassifier(n_neighbors=5)

#treino do modelo
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

#desempennho do modelo
print("\n\nAcurácia nos dados de teste:", accuracy_score(y_test, y_pred))
print("\nF1 nos dados de teste:", f1_score(y_test, y_pred))
print("\nRelatório de Classificação nos dados de teste:\n", classification_report(y_test, y_pred))

#grid search
param_grid = {'n_neighbors': [3, 5, 7, 9],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}

grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Melhores parâmetros:", grid_search.best_params_)

best_knn_model = grid_search.best_estimator_

y_pred = best_knn_model.predict(X_test)

#realiza a validação cruzada com os dados do grid usando o melhor modelo
cv_scores_grid = cross_val_score(best_knn_model, X_normalized, y, cv=5, scoring='accuracy')
print("Acurácia média com cross-validation  usando os dados do grid:", cv_scores_grid.mean())

print("\nF1 nos dados de teste grid:", f1_score(y_test, y_pred))
print("\nRelatório de Classificação nos dados de teste grid:\n", classification_report(y_test, y_pred))

#matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.savefig(os.path.join("knn/", 'matriz_confusao.png'))