import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------
# 1) Función F: calcula el residuo en cada celda
# -----------------------------------------------------
def F(x, w):
    nrows, ncols = x.shape
    fx = np.zeros_like(x)
    for i in range(nrows):
        for j in range(ncols):
            xij = x[i, j]
            x_up = 0 if i == 0 else x[i-1, j]
            x_down = 0 if i == nrows - 1 else x[i+1, j]
            x_left = 1 if j == 0 else x[i, j-1]
            x_right = 0 if j == ncols - 1 else x[i, j+1]
            fx[i, j] = (1/4)*(
                x_up + x_down + x_left + x_right
                - 1/2*(xij*x_down - xij*x_up + w*x_right - w*x_left)
            ) - xij
    return fx

# -----------------------------------------------------
# 2) Jacobiano
# -----------------------------------------------------
def jacobiano(matrix, w):
    nrows, ncols = matrix.shape
    N = nrows * ncols
    J = np.zeros((N, N))
    
    def idx(i, j):
        return i*ncols + j
    
    for i in range(nrows):
        for j in range(ncols):
            fila = idx(i, j)
            xij = matrix[i, j]
            vi1 = 1 if (j == 0) else (1/4)*matrix[i, j-1]
            vi2 = 0 if (j == ncols-1) else matrix[i, j+1]
            col_diag = idx(i, j)
            J[fila, col_diag] = 1/4 * (-1/2 * vi2 + 1/2 * vi1) - 1
            if j > 0:
                col_left = idx(i, j-1)
                J[fila, col_left] = 1/4 * (1 + 1/2*w)
            if j < ncols-1:
                col_right = idx(i, j+1)
                J[fila, col_right] = 1/4 * (1 - 1/2*w)
            if i > 0:
                row_up = idx(i-1, j)
                J[fila, row_up] = 1/4 * (1 + 1/2 * xij)
            if i < nrows-1:
                row_down = idx(i+1, j)
                J[fila, row_down] = 1/4 * (1 - 1/2 * xij)
    
    return J

# -----------------------------------------------------
# 3) Método de Gauss-Seidel para sistemas no lineales
# -----------------------------------------------------
def gauss_seidel_nonlineal(matriz, w=1, tol=1e-10, max_iter=1000):
    start_time = time.time()
    
    for k in range(max_iter):
        Fx = F(matriz, w).reshape(-1)
        normFx = np.linalg.norm(Fx)
        if normFx < tol:
            print(f"✅ Convergencia alcanzada en la iteración {k}, norma={normFx:.2e}")
            break
        
        J = jacobiano(matriz, w)
        N = len(Fx)
        H = np.zeros_like(Fx)
        
        for i in range(N):
            suma = 0
            for j in range(N):
                if j != i:
                    suma += J[i, j] * H[j]
            if abs(J[i, i]) > 1e-14:
                H[i] = (-Fx[i] - suma) / J[i, i]
            else:
                H[i] = 0
        
        matriz = (matriz.reshape(-1) + H).reshape(matriz.shape)
    
    end_time = time.time()
    print(f"🕒 Tiempo total: {end_time - start_time:.4f} segundos")
    print(f"🔁 Iteraciones realizadas: {k + 1}")
    
    return matriz

# -----------------------------------------------------
# 4) Configuración principal
# -----------------------------------------------------
ny, nx = 7, 52
Vx_inicial = np.tile(np.linspace(1, 0, nx), (ny, 2))

# Resolviendo con método de Gauss-Seidel no lineal
Vx_resultante = gauss_seidel_nonlineal(Vx_inicial, w=1)

# -----------------------------------------------------
# 5) Graficar el resultado
# -----------------------------------------------------
plt.figure(figsize=(12, 4))
plt.imshow(Vx_resultante, cmap='viridis', origin='lower', 
           extent=[0, 52, 0, 7], aspect='auto')
plt.colorbar(label='Velocidad Vx')
plt.title('Distribución de la velocidad Vx en la malla (flujo de izq. a der.)')
plt.xlabel('Coordenada x (cm)')
plt.ylabel('Coordenada y (cm)')
plt.show()
