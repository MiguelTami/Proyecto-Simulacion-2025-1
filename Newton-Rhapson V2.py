import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1) Función F: calcula el residuo en cada celda
# -----------------------------------------------------
def F(x, w):
    """
    x: matriz de dimensiones (n_y, n_x) = (7, 52)
       donde x[i,j] es la velocidad Vx en la fila i, columna j.
       i va en dirección vertical (0..6),
       j va en dirección horizontal (0..51).
    w: vorticidad o velocidad en Y (si se usa).
    """
    nrows, ncols = x.shape  # nrows=7, ncols=52
    fx = np.zeros_like(x)
    for i in range(nrows):
        for j in range(ncols):
            xij = x[i, j]
            
            # Vecino de arriba
            if i == 0:
                x_up = 0
            else:
                x_up = x[i-1, j]
            
            # Vecino de abajo
            if i == nrows - 1:
                x_down = 0
            else:
                x_down = x[i+1, j]
            
            # Vecino de la izquierda
            # Forzamos la condición de contorno: en j=0, la celda izquierda se asume 1
            if j == 0:
                x_left = 1  # Flujo de entrada con velocidad 1
            else:
                x_left = x[i, j-1]
            
            # Vecino de la derecha
            if j == ncols - 1:
                x_right = 0
            else:
                x_right = x[i, j+1]
            
            # Mismo residuo que en tu código original
            # (sólo que ahora i es la fila y j la columna)
            fx[i, j] = (1/4)*(
                x_up + x_down + x_left + x_right
                - 1/2*(xij*x_down - xij*x_up + w*x_right - w*x_left)
            ) - xij
    return fx

# -----------------------------------------------------
# 2) Función jacobiano: construye la matriz de derivadas
# -----------------------------------------------------
def jacobiano(matrix, w):
    """
    Construye la matriz jacobiana de tamaño (nrows*ncols, nrows*ncols).
    """
    nrows, ncols = matrix.shape  # 7, 52
    N = nrows * ncols            # 7*52 = 364
    J = np.zeros((N, N))
    
    # Función auxiliar para mapear (i, j) a un índice 1D
    def idx(i, j):
        return i*ncols + j
    
    # Recorremos cada celda (i, j)
    for i in range(nrows):
        for j in range(ncols):
            fila = idx(i, j)  # fila en la matriz jacobiana
            
            # Extraemos valores vecinos para derivar
            xij = matrix[i, j]
            
            # 2.1) Derivada diagonal dF(i,j)/d x(i,j)
            #     Basado en la forma: F(i,j) = 1/4*(arriba+abajo+izq+der - 1/2*(...)) - x(i,j)
            #     => derivada principal ~ -1 + (términos de x(i,j))
            
            #   Ejemplo rápido (simplificado) de la parte - 1/2*(xij*x_down - xij*x_up + w*x_right - w*x_left):
            #   d/dxij [ xij*x_down ] = x_down
            #   d/dxij [ xij*x_up   ] = x_up
            #   etc.
            
            # En tu código original, se hacía:
            # jacobiana[fila, columna] = 1/4 * (-1/2 * vi2 + 1/2 * vi1) - 1
            # para la diagonal, donde vi1, vi2 son vecinos.
            
            # Podemos seguir esa lógica directamente:
            vi1 = 1 if (j == 0) else (1/4)*matrix[i, j-1]  # (equivalente a tu vi1)
            vi2 = 0 if (j == ncols-1) else matrix[i, j+1]  # (equivalente a tu vi2)
            
            col_diag = idx(i, j)
            J[fila, col_diag] = 1/4 * (-1/2 * vi2 + 1/2 * vi1) - 1
            
            # 2.2) Aportaciones de vecinos en la misma fila
            # if i == k => m == k, i == j+1 => etc. (versión vectorizada)
            
            # Vecino (i, j-1): "columna anterior"
            if j > 0:
                col_left = idx(i, j-1)
                # Basado en tu original:
                # jacobiana[fila,columna] = 1/4*(1+1/2*w) si i==j+1
                # Ajustamos la condición: i => j => i
                # Lo que importa es que el término en F depende de x_left
                J[fila, col_left] = 1/4 * (1 + 1/2*w)
            
            # Vecino (i, j+1): "columna siguiente"
            if j < ncols-1:
                col_right = idx(i, j+1)
                J[fila, col_right] = 1/4 * (1 - 1/2*w)
            
            # 2.3) Aportaciones de vecinos en la misma columna (fila arriba/abajo)
            # if i == j => etc. Siguiendo tu lógica:
            
            # Vecino arriba (i-1, j)
            if i > 0:
                row_up = idx(i-1, j)
                # jacobiana[fila, row_up] = 1/4 * (1 + 1/2 * matrix[m,m]) ...
                # En tu caso, "m, m" no es correcto cuando la malla no es cuadrada en i=j.
                # Vamos a simplificar, asumiendo que la dependencia es 1/4*(algo).
                J[fila, row_up] = 1/4 * (1 + 1/2 * xij)
            
            # Vecino abajo (i+1, j)
            if i < nrows-1:
                row_down = idx(i+1, j)
                J[fila, row_down] = 1/4 * (1 - 1/2 * xij)
    
    return J

# -----------------------------------------------------
# 3) Método de Newton-Raphson
# -----------------------------------------------------
def newton_raphson(matriz, iteracciones=100):
    for k in range(iteracciones):
        Fx = F(matriz, 2).reshape(-1)
        normFx = np.linalg.norm(Fx)
        if normFx < 1e-10:
            print(f"Convergencia alcanzada en la iteración {k}, norma={normFx}")
            break
        
        # Jacobiano evaluado con w
        J = jacobiano(matriz, 0)
        
        # En tu código original, calculas: b = J*x - Fx
        x_flat = matriz.reshape(-1)
        Jx = J.dot(x_flat)
        b = Jx - Fx
        
        # Después construyes newmatrix
        newmatrix = np.zeros_like(x_flat)
        for i in range(len(x_flat)):
            # Resta contribuciones fuera de la diagonal
            for j in range(len(x_flat)):
                if i != j:
                    newmatrix[i] -= x_flat[j]*J[i, j]
            newmatrix[i] += b[i]
            # Dividir por la diagonal
            if abs(J[i, i]) > 1e-14:
                newmatrix[i] /= J[i, i]
        
        # Actualizar la solución: x^{(k+1)} = newmatrix
        matriz = newmatrix.reshape(matriz.shape)
    
    return matriz

# -----------------------------------------------------
# 4) Configuración principal
# -----------------------------------------------------
# Definimos la malla como (n_y, n_x) = (7, 52).
#   => 7 filas en vertical (0..40 cm)
#   => 52 columnas en horizontal (0..400 cm)
ny, nx = 7, 52

# Creamos la matriz inicial de velocidades
# Queremos que la columna j=0 sea la "entrada" con V=1
Vx_inicial = np.tile(np.linspace(1, 0, nx), (ny, 1))

# Resolvemos con Newton-Raphson
Vx_resultante = newton_raphson(Vx_inicial, iteracciones=100)

# -----------------------------------------------------
# 5) Graficar el resultado
# -----------------------------------------------------
plt.figure(figsize=(12, 4))
# La matriz es de tamaño (7,52). 
# Cada fila i corresponde a la coordenada Y, cada columna j a la coordenada X.
# extent=[x_min, x_max, y_min, y_max] => [0, 400, 0, 40]
# origin='lower' => la fila i=0 se ubica abajo, i=6 arriba.
plt.imshow(Vx_resultante, cmap='viridis', origin='lower', 
           extent=[0, 400, 0, 40], aspect='auto')
plt.colorbar(label='Velocidad Vx')
plt.title('Distribución de la velocidad Vx en la malla (flujo de izq. a der.)')
plt.xlabel('Coordenada x (cm)')  # de 0 a 400
plt.ylabel('Coordenada y (cm)')  # de 0 a 40
plt.show()

