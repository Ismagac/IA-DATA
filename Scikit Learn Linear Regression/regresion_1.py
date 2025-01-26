import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#1 - Cargar tips y mostrar algo de info
tips = sns.load_dataset('tips')

print("\n Cinco filas:\n", tips.head())
print("\n Información de tips: \n")
print(tips.info())
print("\n Estadísticas: \n", tips.describe())

#2 - Limpieza de datos
print("\n Valores nulos: \n", tips.isnull().sum())
print("Filas duplicadas:", tips.duplicated().sum())

# Se supone que el dataset tips no tiene valores nulos ni duplicados
tips.dropna(inplace=True)
tips.drop_duplicates(inplace=True)

# outliers -  lo hago ccn IQR por si tips es muy asimétrico para hacerlo con cuartiles, si fuese más simétrico, podría hacerlo con Z-score
Q1_tip = tips['tip'].quantile(0.25)
Q3_tip = tips['tip'].quantile(0.75)
IQR_tip = Q3_tip - Q1_tip #este es el IQR

#esto para poner los limits
lower_bound_tip = Q1_tip - 1.5 * IQR_tip
upper_bound_tip = Q3_tip + 1.5 * IQR_tip

before_iqr_rows = tips.shape[0]
# para filtrar por tips
tips = tips[(tips['tip'] >= lower_bound_tip) & (tips['tip'] <= upper_bound_tip)]
after_iqr_rows = tips.shape[0]
print(f"\n IQR en la columna tip \n")
print(f"FIlas antes de IQR : {before_iqr_rows}")
print(f"Filas después de IQR: {after_iqr_rows}")

#3 - EDA
#univariante - Histograma
plt.hist(tips['total_bill'], bins=20, color='purple', edgecolor='black')
plt.title("Histograma de 'total_bill'")
plt.xlabel("Total Bill")
plt.ylabel("Frecuencia")
plt.savefig("histograma.png")
plt.close()

#bivariante - Dispersion
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', palette='Purples')
plt.title("Dispersion total_bill - tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.savefig("dispersion.png")
plt.close()

# multivariant - Heatmap de correlación
numeric_data = tips.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="Purples")
plt.title("Mapa de correlación")
plt.savefig("mapacalor.png")
plt.close()
# en un entorno Jupyter, podría usar plt.show() en vez de plt.savefig() y ver todo en una ventana con flechas.

# 4 - Preparación de datos	
# para tranformar a datos numéricos, drop first porque en el caso del día aunque borres el primero, si el resto son 0, se sabe que el primero es el correcto, como si fuese por descarte
tips_dummies = pd.get_dummies(tips, columns=['sex','smoker','day','time'], drop_first=True)

# para las características y la variable objetivo
X = tips_dummies.drop('tip', axis=1)
y = tips_dummies['tip']

# split de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=11
    #para el random state, si no se pone, cada vez que se ejecute, se obtienen resultados diferentes o si pongo NONE
)

# para escalar las variables numéricas
scaler = StandardScaler()
num_cols = ['total_bill', 'size']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
#para la prueba sin fit, los datos de prueba no los tiene que ver el modelo
X_test[num_cols] = scaler.transform(X_test[num_cols])

#5-  Modelado
# he utilizado este par, uno lineal y otro de arbol de decisión
modelos = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(random_state=11)
    #Igual que antes, para el random state, si no se pone, cada vez que se ejecute, se obtienen resultados diferentes o si pongo NONE
}

for nombre_modelo, modelo in modelos.items():
    #se entrena el modelo con los datos del entrenamiento
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    #Seria asi, pero me da fallo por la V. de Scikit-learn en mi entorno: srmse = mean_squared_error(y_test, y_pred, squared=False)
    #Asi es de manera manual
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n Modelo: {nombre_modelo}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.2f}")

#  Modelo: LinearRegression
#   MAE  : 0.56
#   RMSE : 0.72
#   R²   : 0.43

#  Modelo: RandomForestRegressor
#   MAE  : 0.51
#   RMSE : 0.75
#   R²   : 0.37

# Estos son los resultados que he obtenido en mi entorno, en este caso dependiendo de las métricas que se quieran priorizar, pero en general pese a que en el modelo
# de regresión lineal el MAE para los errores absolutos pequeños es mayor, el RMSE para errores grandes es menor, y el R² se acerca más a 1.
# Es decir, que en general, el modelo de regresión lineal es mejor que el de arbol de decisión, aunque ambos son relativamente malos, el R² es bajo en ambos casos.