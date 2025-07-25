import numpy as np
import matplotlib.pyplot as plt

#Funcion f
def F(x,w):
  fx=np.zeros((x.shape[0],x.shape[1]))
  for i in range (x.shape[0]):
    #For de filas
    for j in range (x.shape[1]):
      #For de columnas
      xij=x[i,j]
      if i==0:
        x1i=0
      else:
        x1i=x[i-1,j]
      if i==x.shape[0]-1:
        xi1=0
      else:
        xi1=x[i+1,j]
      if j==0:
        x1j=1
      else:
        x1j=x[i,j-1]
      if j==x.shape[1]-1:
        xj1=0
      else:
        xj1=x[i,j+1]
      fx[i,j]=1/4*(xi1+x1i+xj1+x1j-1/2*(xij*xi1-xij*x1i+w*xj1-w*x1j))-xij
  return fx

def jacobiano (matrix, w):
  jacobiana=np.zeros((matrix.size,matrix.size))
  fila=-1
  diagonal=0
  for m in range (matrix.shape[0]):
    for i in range ( matrix.shape[1]):
      fila+=1
      columna=-1
      for k in range (matrix.shape[0]):
        for j in range (matrix.shape[1]):
          columna+=1
          if m==k and i==j:
            if j==0:
              vi1=1
            else:
              vi1=1/4*matrix[k,j-1]
            if j==matrix.shape[1]-1:
              vi2=0
            else:
              vi2=matrix[k,j+1]
            jacobiana[fila,columna]=1/4*(-1/2*vi2+1/2*vi1)-1
          if m==k:
            if i==j+1:
              #Columna anterior
              jacobiana[fila,columna]=1/4*(1+1/2*w)
            if i==j-1:
              #Columna siguiente
              jacobiana[fila,columna]=1/4*(1-1/2*w)
          if i==j:
            if m==k+1:
              #Fila anterior
              jacobiana[fila,columna]=1/4*(1+1/2*matrix[m,m])
            if m==k-1:
              #Fila siguiente
              jacobiana[fila,columna]=1/4*(1-1/2*matrix[m,m])


  return jacobiana

def newton_raphson(matriz,iteracciones=100):
  for k in range (iteracciones):
    Fx=F(matriz,1).reshape(matriz.size)
    if (np.linalg.norm(Fx)<1e-10):
      break
    Jac=jacobiano(matriz,0)
    Jxi=np.dot(Jac,matriz.reshape(matriz.size))
    b=Jxi-Fx
    xi=matriz.reshape(matriz.size)
    newmatrix=np.zeros(matriz.size)
    for i in range (matriz.size):
      for j in range (matriz.size):
        if not(i==j):
          newmatrix[i]-=xi[j]*Jac[i,j]
      newmatrix[i]+=b[i]
      newmatrix[i]/=Jac[i,i]
    newmatrix=newmatrix.reshape(matriz.shape)
    matriz=newmatrix
  return matriz

# Para fines de demostración, generaremos una matriz de ejemplo:
vx_matrix = newton_raphson(np.zeros((40, 400)), 100)

# Crear un heatmap que muestre la distribución de la velocidad Vx
plt.figure(figsize=(10, 6))
# 'extent' define las dimensiones físicas de la malla (en cm)
plt.imshow(vx_matrix, cmap='viridis', origin='lower', extent=[0, 400, 0, 40])
plt.colorbar(label='Velocidad Vx')
plt.title('Distribución de la velocidad Vx en la malla')
plt.xlabel('Coordenada x (cm)')
plt.ylabel('Coordenada y (cm)')
plt.show()


