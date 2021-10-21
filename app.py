
from pandas.io import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import pickle


datos=pd.read_excel("Gastos.xlsx")

###################################################################################################################################

st.subheader('INTEGRANTES:')
st.subheader('- Tatiana Chavez Perez')
st.subheader('- Carlos Fernando Gonzalez Sarmiento')
st.subheader('- Harold Rojas Camacho')

st.title('Valor salario')

st.write('Esta app estima el valor de un salario partiendo de características específicas, como por ejemplo, las deudas, el puesto en el trabajo, sexo, edad, egresos,entre otros. Para empezar se hará un pequeña descripción estadística de la base de datos tomada:')

st.subheader('Análisis descriptivo')
st.write(datos)
st.write('Encabezado de la base de datos')
st.write(datos.head())

st.write('Descripción de la base de datos:')
st.write(datos.describe())

#############################################
st.write('De acuerdo al siguiente gráfico se observa que hay más hombres que mujeres con unos porcentajes de 53,4% para hombres y 45.8% para mujeres')
x = datos['SEXO']=="M"
z1= x.sum()/len(datos['SEXO'])
x1 = datos['SEXO']=="F"
z2= x1.sum()/len(datos['SEXO'])
fig, ax = plt.subplots()
labels = ["Hombres","Mujeres"]
z=[z1,z2]
ax.pie(z, labels=labels, autopct='%1.1f%%',colors= ['blue','pink'])
ax.set_title('Porcentaje de hombres y mujeres')
st.pyplot(fig)

##########################################################
st.write('Se puede observar en la gráfica que gran parte de los trabajadores tienen a cargo 2 personas, seguido de 3 personas y una sola persona')
fig, ax = plt.subplots()
ax.bar(datos['Personas'].value_counts().index,datos['Personas'].value_counts())
ax.set_title('Acumulado de persona a cargo')
ax.set_xlabel('Numero de personas a cargo')
st.pyplot(fig)

#####################################################
st.write('Gracias al gráfico podemos notar que la mitad de los trabajadores no cuenta con una ayuda externa')

x = ['Sí','No'] 
c = [(datos['Ayuda']=='Sí').sum(),(datos['Ayuda']=='No').sum()]
fig,ax= plt.subplots()
ax.bar(x,c, color='tab:purple')
ax.set_title("¿Recibe alguna ayuda?") 
ax.set_ylabel("Frecuencia")
st.pyplot(fig)
########################################################
st.write('A continuación se presenta la frecuencia de número de hijos donde se puede evidenciar que una gran parte tiene 2 hijos, seguido de un hijo y por último que no tienen hijos')
fig,ax= plt.subplots()
ax.bar(datos['Hijos'].value_counts().index,datos['Hijos'].value_counts())
ax.set_title('Acumulado de hijos')
ax.set_xlabel('Numero de hijos')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)
st.write(' ')

##############################################################
st.write('Respecto a la escolaridad se puede evidenciar que del 56.2% de los trabajadores son universitarios o tiene un pregrado, seguido de posgrado y técnico, tan solo el 1.5% tienen el título de bachiller')

fig,ax= plt.subplots()
colores = ['blue','pink','red','green']
act = ["Universitario","Posgrado","Técnico","Secundaria"]
edu = datos['Escolaridad'].value_counts()
ax.pie(edu,autopct = '%1.1f%%',labels=act,colors=colores) 
st.pyplot(fig)
##########################################################
st.write('En el siguiente gráfico se puede observar que el nivel con más frecuencia es el 3, seguido del nivel 2, nivel 4 y por último nivel 1.')
fig,ax= plt.subplots()
s = datos['Nivel'].value_counts().index
m = datos['Nivel'].value_counts()
ax.bar(s,m,color='tab:blue',width=0.45)
ax.set_title("Nivel")
ax.set_ylabel("Frecuencia")
##########################################################
st.write('Se puede ver las variaciones que tuvo la prueba de conocimientos respecto a cada trabajador')
fig,ax= plt.subplots()
ax.plot(datos['Conocimientos'])
st.pyplot(fig)

##########################################################
st.write('')
st.write('GRÁFICO DE DISPERSIÓN DE EDAD CONTRA EL SALARIO')
fig,ax= plt.subplots()
ax.scatter(datos['EDAD'],datos['Salario'])
st.pyplot(fig)
#########################################################
st.write('Correlaciones entre variables por el método de Pearson')
st.write(datos.corr(method='pearson'))

###################################################################################################################################

#Prediccion del modelo

st.subheader('Variables del modelo')

Edad=st.number_input('Edad',min_value=0)
Hijos=st.number_input('Numero de hijos',min_value=0)
Personas=st.number_input('Numero de personas de las que esta a cargo',min_value=0)
Egresos=st.number_input('La cantidad de egresos que tiene',min_value=0)
Deudas=st.number_input('Numero de deudas pendientes',min_value=0)
Conocimientos=st.number_input('Puntaje en examen de conocimientos',min_value=0)
Sexo=st.selectbox('Sexo',options=['Masculino','Femenino'])
Escolaridad=st.selectbox('Nivel educativo maximo alcanzado',options=['Secundaria','Posgrado','Tecnico','Universitario'])
Nivel=st.selectbox('Nivel de trabajo',options=['1','2','3','4'])
Estrato=st.selectbox('Estrato',options=['1','2','3','4','5'])
Ayuda=st.selectbox('¿Ha recibido alguna ayuda economica?',options=['no','si'])
Situacion=st.selectbox('¿En que tipo de situacion economica cree que está?',options=['Aceptable','Buena','Regular','Dificil'])


if Sexo=='Femenino':
    Sexo_F=1
else:
    Sexo_F=0

if Escolaridad=='Posgrado':
    Escolaridad_Posgrado=1
    Escolaridad_Técnico=0
    Escolaridad_Universitario=0
elif Escolaridad=='Tecnico':
    Escolaridad_Posgrado=0
    Escolaridad_Técnico=1
    Escolaridad_Universitario=0
elif Escolaridad=='Universitario':
    Escolaridad_Posgrado=0
    Escolaridad_Técnico=0
    Escolaridad_Universitario=1
else:
    Escolaridad_Posgrado=0
    Escolaridad_Técnico=0
    Escolaridad_Universitario=0   



if Estrato=='2':
    Estrato_2=1
    Estrato_3=0
    Estrato_4=0
    Estrato_5=0
elif Estrato=='3':
    Estrato_2=0
    Estrato_3=1
    Estrato_4=0
    Estrato_5=0
elif Estrato=='4':
    Estrato_2=0
    Estrato_3=0
    Estrato_4=1
    Estrato_5=0
elif Estrato=='5':
    Estrato_2=0
    Estrato_3=0
    Estrato_4=0
    Estrato_5=1
else:
    Estrato_2=0
    Estrato_3=0
    Estrato_4=0
    Estrato_5=0


if Nivel=='2':
    Nivel_2=1
    Nivel_3=0
    Nivel_4=0
elif Nivel=='3':
    Nivel_2=0
    Nivel_3=1
    Nivel_4=0
elif Nivel=='4':
    Nivel_2=0
    Nivel_3=0
    Nivel_4=1
else:
    Nivel_2=0
    Nivel_3=0
    Nivel_4=0

if Ayuda=='si':
    Ayuda_si=1
else:
    Ayuda_si=0

if Situacion=='Buena':
    Situacion_Buena=1
    Situacion_Dificil=0
    Situacion_Regular=0
elif Situacion=='Dificil':
    Situacion_Buena=0
    Situacion_Dificil=1
    Situacion_Regular=0
elif Situacion=='Regular':
    Situacion_Buena=0
    Situacion_Dificil=0
    Situacion_Regular=1
else:
    Situacion_Buena=0
    Situacion_Dificil=0
    Situacion_Regular=0


rf_pickle=open('rf_reg.pickle','rb')

rf=pickle.load(rf_pickle)

rf_pickle.close()

st.subheader('Predicción salario')

pred_rf=rf.predict([[Edad,Hijos,Personas,Egresos,Deudas,Conocimientos,Sexo_F,Escolaridad_Posgrado,Escolaridad_Técnico,Escolaridad_Universitario,Nivel_2,Nivel_3,Nivel_4,Estrato_2,Estrato_3,Estrato_4,Estrato_5,Ayuda_si,Situacion_Buena,Situacion_Dificil,Situacion_Regular]])


st.write(f'Se estima que su salario será de aproximadamente: ${pred_rf[0]}')
