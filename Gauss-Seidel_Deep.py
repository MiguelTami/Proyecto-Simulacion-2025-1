import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------------
# 1. Función de resolución por Gauss-Seidel
# -----------------------------------------------------
def gauss_seidel_solver(Vx_initial, w=2, max_iter=1000, tol=1e-5):
    Vx = Vx_initial.copy()
    nrows, ncols = Vx.shape
    errors = []  # Para almacenar la historia de convergencia
    start_time = time.time()
    
    for it in range(max_iter):
        max_error = 0.0
        for i in range(nrows):
            for j in range(ncols):
                # Condiciones de frontera fijas
                if j == 0:
                    Vx_old = 1.0  # Entrada izquierda V=1
                elif j == ncols-1:
                    Vx_old = 0.0  # Salida derecha V=0
                else:
                    Vx_old = Vx[i, j]
                
                # Cálculo de vecinos (actualizando con valores nuevos cuando sea posible)
                Vx_up = Vx[i-1, j] if i > 0 else 0.0
                Vx_down = Vx[i+1, j] if i < nrows-1 else 0.0
                Vx_left = 3.0 if j == 0 else Vx[i, j-1]  # Usa valor ya actualizado si j>0
                Vx_right = 0.0 if j == ncols-1 else Vx[i, j+1]
                
                # Ecuación principal (modificada para mejor convergencia)
                Vx_new = (1/4)*(Vx_up + Vx_down + Vx_left + Vx_right) - (1/8)*(Vx_old*(Vx_down - Vx_up) + w*(Vx_right - Vx_left))
                
                # Actualizar solo si no es condición de frontera
                if j not in [0, ncols-1]:
                    Vx[i, j] = Vx_new
                    current_error = abs(Vx_new - Vx_old)
                    if current_error > max_error:
                        max_error = current_error
        
        errors.append(max_error)
        
        # Criterio de parada
        if max_error < tol:
            break
    
    elapsed_time = time.time() - start_time
    print(f"Convergencia en {it+1} iteraciones")
    print(f"Tiempo total: {elapsed_time:.4f} s")
    print(f"Error final: {max_error:.2e}")
    
    return Vx, errors

# -----------------------------------------------------
# 2. Configuración inicial
# -----------------------------------------------------
ny, nx = 7, 52  # 7 filas (vertical), 52 columnas (horizontal)

# Condición inicial: perfil lineal de velocidades (1 en izquierda, 0 en derecha)
Vx_inicial = np.zeros((ny, nx))
Vx_inicial[:, 0] = 1.0  # Velocidad 1 en la entrada izquierda

# Parámetros del solver
w = 2.0       # Parámetro de vorticidad
tolerancia = 1e-5
max_iteraciones = 1000

# -----------------------------------------------------
# 3. Ejecutar el solver
# -----------------------------------------------------
Vx_resultado, errores = gauss_seidel_solver(
    Vx_initial=Vx_inicial,
    w=w,
    max_iter=max_iteraciones,
    tol=tolerancia
)

# -----------------------------------------------------
# 4. Graficar resultados
# -----------------------------------------------------
plt.figure(figsize=(15, 5))

# Gráfica 1: Campo de velocidades
plt.subplot(1, 2, 1)
plt.imshow(Vx_resultado, cmap='viridis', origin='lower', extent=[0, 400, 0, 40], aspect='auto')
plt.colorbar(label='Velocidad Vx (m/s)')
plt.title('Campo de velocidades Vx')
plt.xlabel('Posición horizontal (cm)')
plt.ylabel('Posición vertical (cm)')

# Gráfica 2: Convergencia
plt.subplot(1, 2, 2)
plt.semilogy(errores, 'r-', linewidth=2)
plt.title('Convergencia del método Gauss-Seidel')
plt.xlabel('Iteración')
plt.ylabel('Error máximo (escala log)')
plt.grid(True)

plt.tight_layout()
plt.show()