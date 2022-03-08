import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#np.random.seed(10)
#BD
bd=(pd.read_csv('covid_mexico_final.csv')).drop(columns=['Unnamed: 0'])
min_max_scaler=preprocessing.MinMaxScaler()
BD=min_max_scaler.fit_transform(bd)
#%%Sample
sample_size=1000000
sample_numbers=np.random.randint(len(BD),size=sample_size)
B_D=np.zeros([sample_size,16])
fila=0
for i in sample_numbers:
 B_D[fila,:]=(BD[i,:])
 fila+=1
#%%
train, test = (train_test_split(BD, test_size=0.3))
#Variables
L=10 #número de neuronas en la capa oculta (modificable, siempre mayor a 2, 
entero)
N=15 #número de entradas 
M=1 #número de salidas 
alfa=1 #valor ajustable
A=1 #valor ajustable
E=1
x=train[:,0:15] #entradas
d=train[:,[15]] #salidas deseadas
Wh=np.reshape(np.random.random((L*N)),(L,N)) #pesos ocultos
Wo=np.reshape(np.random.random((M*L)),(M,L)) #pesos de salida

#%%
#Aprendizaje
COUNT=0 #contador de vueltas que hace el while
while E>0.00001:
 COUNT+=1
 #print(E)
 for filas in range(len(x)):
 
 #Forward
 neth=Wh@(np.reshape(x[filas],(N,1))) #en la fórmula viene un x 
transpuesta, aquí lo hago dentro del reshape
 #tomo el vector del renglón 
[0,1,... filas] (un renglón cada vuelta) y lo transpongo ahora en forma N,1
 yh=1/(1+np.exp(-A*(neth)))
 neto=Wo@yh
 yo=1/(1+np.exp(-A*(neto)))
 
 #Backward
 deltao=(np.reshape(d[filas],(M,1))-yo)*yo*(1-yo) #aplico nuevamente 
reshape para hacer la transpuesta, toda esta operación es directa (nada 
matricial)
 deltah=yh*(1-yh)*((np.transpose(Wo))@deltao) #tmb pude haber puesto 
Wo.T porque tmb es transponer
 
 #Carrusel y cálculo de deltWh y deltaWo
 Wo+=alfa*deltao@(np.transpose(yh)) 
 Wh+=alfa*deltah@(np.reshape(x[filas],(1,N))) #aquí aunque ya la fila 
de x es del tamaño que quiero y no necesito transponer, sí necesito el 
reshape pq me lo toma como un array trunco de (,4) en lugar de (1,4)
 
 E=max(np.abs(deltao))
 print(E)
 
#%%
#Estimaciones
#X_desconocida=np.array([[1,0,1,1,0.2231,1,1,1,1,1,1,1,1,1,1],[1,0,1,1,0.2470
,1,1,1,1,1,1,1,1,1,0],[0,0,1,1,0.2396,1,1,1,1,1,1,1,1,1,1]])
X_desconocida=test[:,0:15] #entradas
Y_estimadas=np.zeros((len(X_desconocida),M)) #Matriz de ceros de longitud de 
la matriz que busco estimar, por las salidas que busco
for filas in range(len(X_desconocida)): 
#Forward
 neth=Wh@(np.reshape(X_desconocida[filas],(N,1)))
 yh=1/(1+np.exp(-A*(neth)))
 neto=Wo@yh
 yo=1/(1+np.exp(-A*(neto)))
 
 Y_estimadas[filas,:]=yo.T
#%%
#accuracy
Y_estimadas_round=np.round(Y_estimadas)
Y_reales=test[:,[15]]
a=0
 
for i in range(len(Y_estimadas_round)):
 a+=sum(Y_estimadas_round[i] == Y_reales[i])
 
accuracy=a/len(Y_estimadas_round)
 
print('Para una A de',A,', una alfa de',alfa,'y',L,'neuronas')
print('La cuenta fue de',COUNT)
print('El accuracy fue de',accuracy)
