
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import numpy as np
import pickle

pd.options.mode.chained_assignment = None


#Importar datos
datos=pd.read_excel("Gastos.xlsx")

del(datos['ESTADO CIVIL'])
Sexo = pd.get_dummies(datos['SEXO'],prefix = 'Sexo')
Escolaridad= pd.get_dummies(datos['Escolaridad'],prefix = 'Escolaridad')
Nivel = pd.get_dummies(datos['Nivel'],prefix = 'Nivel')
Estrato = pd.get_dummies(datos['Estrato'],prefix = 'Estrato')
Ayuda=pd.get_dummies(datos['Ayuda'],prefix = 'Ayuda')
Situacion= pd.get_dummies(datos['Situación'],prefix = 'Situacion')

datos1 = datos.drop(['SEXO','Escolaridad','Nivel','Estrato','Ayuda','Situación'],axis=1)
datos12 = pd.concat([datos1,Sexo],axis = 1)
datos14=pd.concat([datos12,Escolaridad],axis = 1)
datos15 = pd.concat([datos14,Nivel],axis = 1)
datos16 = pd.concat([datos15,Estrato],axis = 1)
datos17= pd.concat([datos16,Ayuda],axis = 1)
datos18 = pd.concat([datos17,Situacion],axis = 1)
datos19=datos18.rename(columns={'Nivel_2.0':'Nivel_2','Nivel_3.0':'Nivel_3','Nivel_4.0':'Nivel_4','Estrato_2.0':'Estrato_2','Estrato_3.0':'Estrato_3','Estrato_4.0':'Estrato_4','Estrato_5.0':'Estrato_5','Ayuda_Sí':'Ayuda_si','Nivel_4.0':'Nivel_4'})

datos2=datos19.drop(['Sexo_M','Escolaridad_Secundaria','Nivel_1.0','Estrato_1.0','Ayuda_No','Situacion_Aceptable'],axis=1)
datos21=  datos2[datos2['EDAD'].notna()]
datos21[['Hijos']]=datos[['Hijos']].fillna(0)
datos21[['Personas']]=datos[['Personas']].fillna(0)
datos21[['Conocimientos']]=datos[['Conocimientos']].fillna(0)

datos22=  datos21[datos21['Salario'].notna()] 
datosfinal=datos22[datos22['Egresos'].notna()]


X=datosfinal.drop(['Salario'],axis=1)
y=datosfinal['Salario']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

param_grid={'n_estimators':[50,100,150],
            'max_features':[3,5,7],
            'max_depth':[3,4,5]}


grid_search=GridSearchCV(RandomForestRegressor(),param_grid,cv=5)
grid_search.fit(X_train,y_train)

reg_rf=RandomForestRegressor(**grid_search.best_params_)

reg_rf.fit(X_train,y_train)

y_train_fit=reg_rf.predict(X_train)
y_pred=reg_rf.predict(X_test)

print(f'El error cuadrático medio del modelo es: {mean_absolute_error(y_train,y_train_fit)}')
print(f'El Error absoluto medio del modelo es: {mean_absolute_error(y_test,y_pred)}')

 #Se decidió no aplicar el logaritmo sobre el salario para predecir el salario exacto, pero las metricas mejoran al aplicar el logaritmo

rf_pickle=open('rf_reg.pickle','wb')

pickle.dump(reg_rf,rf_pickle)

rf_pickle.close()

