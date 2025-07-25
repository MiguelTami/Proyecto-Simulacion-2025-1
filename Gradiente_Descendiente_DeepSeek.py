import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------
# 1. Función F (sin cambios)
# -----------------------------------------------------
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
            
            fx[i, j] = (1/4)*(x_up + x_down + x_left + x_right) - (1/8)*(xij*(x_down - x_up) + w*(x_right - x_left)) - xij
    return fx

# -----------------------------------------------------
# 2. Jacobiano (sin cambios)
# -----------------------------------------------------
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
            
            # Término diagonal
            J[fila, fila] = (1/4)*(-0.5*(matrix[i+1,j] - matrix[i-1,j]) if 0 < i < nrows-1 else 0) - 1
            
            # Vecinos verticales
            if i > 0:
                J[fila, idx(i-1, j)] = (1/4)*(1 + 0.5*xij)
            if i < nrows-1:
                J[fila, idx(i+1, j)] = (1/4)*(1 - 0.5*xij)
                
            # Vecinos horizontales
            if j > 0:
                J[fila, idx(i, j-1)] = (1/4)*(1 + 0.5*w)
            if j < ncols-1:
                J[fila, idx(i, j+1)] = (1/4)*(1 - 0.5*w)
    return J

# -----------------------------------------------------
# 3. Gradiente Descendente Modificado
# -----------------------------------------------------
def gradiente_descendente(matrix_inicial, w=2, max_iter=50000, tol=1e-5, alpha=0.0001):
    matrix = matrix_inicial.copy()
    nrows, ncols = matrix.shape
    errores = []
    start_time = time.time()
    
    for k in range(max_iter):
        Fx = F(matrix, w)
        normFx = np.linalg.norm(Fx)
        errores.append(normFx)
        
        if normFx < tol:
            break
            
        J = jacobiano(matrix, w)
        gradiente = J.T @ Fx.flatten()
        matrix = matrix - alpha * gradiente.reshape(matrix.shape)
        
    elapsed_time = time.time() - start_time
    
    print(f"=== Resultados Gradiente Descendente ===")
    print(f"Iteraciones realizadas: {k+1}/{max_iter}")
    print(f"Tiempo total: {elapsed_time:.2f} segundos")
    print(f"Norma residual final: {normFx:.2e}")
    
    return matrix, errores

# -----------------------------------------------------
# 4. Configuración y ejecución
# -----------------------------------------------------
# Parámetros comunes
ny, nx = 7, 52
Vx_inicial = np.zeros((ny, nx))
Vx_inicial[:, 0] = 1.0  # Condición inicial

# Parámetros específicos gradiente
parametros_gd = {
    "w": 2.0,
    "max_iter": 20000,
    "tol": 1e-4,
    "alpha": 0.00001  # Paso de aprendizaje reducido para estabilidad
}

# Ejecutar
Vx_gd, errores_gd = gradiente_descendente(Vx_inicial, **parametros_gd)

# -----------------------------------------------------
# 5. Graficar resultados
# -----------------------------------------------------
plt.figure(figsize=(15, 5))

# Campo de velocidades
plt.subplot(1, 2, 1)
plt.imshow(Vx_gd, cmap='viridis', origin='lower', extent=[0,400,0,40], aspect='auto')
plt.colorbar(label='Vx [m/s]')
plt.title('Campo de velocidades (Gradiente Descendente)')
plt.xlabel('Posición horizontal (cm)')
plt.ylabel('Posición vertical (cm)')

# Curva de convergencia
plt.subplot(1, 2, 2)
plt.semilogy(errores_gd, 'r-')
plt.title('Convergencia del Gradiente Descendente')
plt.xlabel('Iteración')
plt.ylabel('Norma del residuo (log)')
plt.grid(True)

plt.tight_layout()
plt.show()  