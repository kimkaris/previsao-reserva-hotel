import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

data = pd.read_csv("Hotel Reservations.csv")

#colunas relevantes
selected_columns = [
    'no_of_weekend_nights', 'no_of_week_nights', 'avg_price_per_room',
    'lead_time', 'arrival_month', 'arrival_date', 'no_of_adults',
    'no_of_children', 'booking_status', 'type_of_meal_plan',
    'required_car_parking_space', 'room_type_reserved', 'arrival_year',
    'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled','no_of_special_requests'
]
data = data[selected_columns]

#tratando as variáveis categóricas
data_tratado = pd.get_dummies(data, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status'], drop_first = True)

# Divide os dados em conjunto de treinamento e teste
X = data_tratado.drop('booking_status_Not_Canceled', axis=1)
y = data_tratado['booking_status_Not_Canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(data_tratado.info())

#cria e treina o modelo
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#avaliando o modelo
classification_rep = classification_report(y_test, y_pred)
print(f'Acurácia:', accuracy_score(y_test, y_pred))
print("\nF1 nos dados de teste:", f1_score(y_test, y_pred))
print(f'\n\nRelatório de classificação:\n{classification_rep}')

#grid search
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = grid_search.best_estimator_
y_pred_grid = best_model.predict(X_test)

#avaliando com o melhor modelo
conf_matrix_grid = confusion_matrix(y_test, y_pred_grid)
classification_rep_grid = classification_report(y_test, y_pred_grid)
cv_scores_grid = cross_val_score(best_model, X_train, y_train, cv=10, scoring='accuracy')
print(f'\n\nAcurácia (Grid Search e média cross validation):',cv_scores_grid.mean())
print("\nF1 nos dados de teste (Grid Search):", f1_score(y_test, y_pred_grid))
print(f'\n\nRelatório de Classificação (Grid Search):\n{classification_rep_grid}')

#matriz de confusão
sns.heatmap(conf_matrix_grid, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsões')
plt.ylabel('Valores reais')
plt.title('Matriz de confusão')
# plt.show()
plt.savefig(os.path.join("gradient_boosting/", 'matriz_confusao.png'))

#importância das características
feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.xlabel('Importância das características')
plt.ylabel('Características')
plt.title('Importância das características no modelo')
# plt.show()
plt.savefig(os.path.join("gradient_boosting/", 'importancia_caracteristicas.png'))