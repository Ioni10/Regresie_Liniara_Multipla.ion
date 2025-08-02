#Scriem biblotecile
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

#scriem independenta
x = [
[1, 20, 1.2],
[2, 35, 1.4],
[3, 60, 1.6],
[4, 80, 1.6],
[5, 120, 1.8],
[6, 150, 2.0],
[7, 180, 2.0],
]
#scriem dependenta
y= np.array([12000,11000,9500,8500,7000,6000,5000])
# Am creat un model de linear regression am fituit valorile lui x pe y
model = LinearRegression()
model.fit(x,y)
fig = plt.figure()
ax = plt.axes(projection="3d") #Am setat 3d
axa_y = [1,2,3,4,5,6,7] #Vechime
axa_x = [20,35,60,80,120,150,180] #km
axa_z = [12000, 11000,9500,8500,7000,6000,5000] # Pret           #am separat caracteristicile
y_real = y #ii spunem programului care este y realul
y_pred = model.predict(x) # aici ii spunem care este predictia lui y
ax.scatter(axa_x ,axa_y , axa_z, color ='blue', label='pret') #aici plotam preturile reale
ax.scatter(axa_x, axa_y, y_pred, color='red', label='predictie pret') # aici plotam predictia preturilor
ax.set_xlabel("Km(mii)") # Scriem specificatii/pretul pe grafic
ax.set_ylabel("Vechimea(ani)")
ax.set_zlabel("Pret(euro)")

plt.title("Regresie Liniara Multipla") # titlul regresiei
plt.legend() # Legenda (explicatiile)
plt.show() # Ne arata graficul

r2= r2_score(y_real , y_pred) # Scorul R^2 care ne arata cat de aproape suntem cu prezicerea de realitate
pret = model.predict([[1, 20 , 1.2]])
pret2 = model.predict([[4, 75 ,1.6]]) # Cerinta de calculat 4 ani cu 75k km si motor 1.6

print("Coeficienti beta1 , beta2",model.coef_)
print("interpretul , beta0 :", model.intercept_)
print("Coeficient R^2", r2)
print("Care ar fi pretul primului model", pret)
print("Calcul masina cu 4ani vechime , 75k km si 1.6 motor", pret2)

