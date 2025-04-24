import numpy as np
import matplotlib.pyplot as plt

def F(x, w):
    nrows, ncols = x.shape
    fx = np.zeros_like(x)
    for i in range(nrows):
        for j in range(ncols):
            xij = x[i, j]
            x_up = x[i-1, j] if i > 0 else 0
            x_down = x[i+1, j] if i < nrows-1 else 0
            x_left = 1 if j == 0 else x[i, j-1]
            x_right = 0 if j == ncols-1 else x[i, j+1]
            
            fx[i, j] = (1/4)*(x_up + x_down + x_left + x_right - 0.5*(xij*x_down - xij*x_up + w*x_right - w*x_left)) - xij
    return fx

def jacobiano(matrix, w):
    nrows, ncols = matrix.shape
    N = nrows * ncols
    J = np.zeros((N, N))
    
    def idx(i, j):
        return i * ncols + j
    
    for i in range(nrows):
        for j in range(ncols):
            fila = idx(i, j)
            xij = matrix[i, j]
            x_up = matrix[i-1, j] if i > 0 else 0
            x_down = matrix[i+1, j] if i < nrows-1 else 0
            
            # Término diagonal
            J[fila, fila] = (1/4)*(-0.5*(x_down - x_up)) - 1
            
            # Vecinos arriba y abajo
            if i > 0:
                J[fila, idx(i-1, j)] = (1/4)*(1 + 0.5*xij)
            if i < nrows-1:
                J[fila, idx(i+1, j)] = (1/4)*(1 - 0.5*xij)
            # Vecinos izquierda y derecha
            if j > 0:
                J[fila, idx(i, j-1)] = (1/4)*(1 + 0.5*w)
            if j < ncols-1:
                J[fila, idx(i, j+1)] = (1/4)*(1 - 0.5*w)
    return J

def gradiente_descendente(matrix_inicial, w=2, max_iter=10000, tol=1e-6, alpha=0.001):
    matrix = matrix_inicial.copy()
    for k in range(max_iter):
        Fx = F(matrix, w).flatten()
        normFx = np.linalg.norm(Fx)
        if normFx < tol:
            print(f"Convergencia en iteración {k}, norma={normFx:.2e}")
            break
        J = jacobiano(matrix, w)
        gradiente = J.T @ Fx  # J^T * F
        matrix = (matrix.flatten() - alpha * gradiente).reshape(matrix.shape)
        if k % 100 == 0:
            print(f"Iter {k}: Norma={normFx:.2e}")
    else:
        print(f"No convergió en {max_iter} iteraciones. Norma={normFx:.2e}")
        
    return matrix
    

# Configuración
ny, nx = 7, 52
Vx_inicial = np.tile(np.linspace(1, 0, nx), (ny, 1))

# Ejecutar
Vx_resultado = gradiente_descendente(Vx_inicial, alpha=0.001)

# Graficar
plt.figure(figsize=(12, 4))
plt.imshow(Vx_resultado, cmap='viridis', origin='lower', extent=[0, 400, 0, 40], aspect='auto')
plt.colorbar(label='Velocidad Vx')
plt.title('Distribución de Velocidad usando Gradiente Descendente')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()