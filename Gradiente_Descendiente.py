import numpy as np
import matplotlib.pyplot as plt

# Reutilizamos la función F del código original para calcular el residuo
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
            
            # Residuo según las ecuaciones de Navier-Stokes discretizadas
            fx[i, j] = (1/4)*(
                x_up + x_down + x_left + x_right
                - 1/2*(xij*x_down - xij*x_up + w*x_right - w*x_left)
            ) - xij
    return fx

# Nueva función para calcular el gradiente numérico de la función objetivo
def calcular_gradiente(matriz, w, h=1e-6):
    """
    Calcula el gradiente de la función objetivo mediante diferencias finitas.
    La función objetivo es la suma de los cuadrados de los residuos.
    
    Args:
        matriz: matriz de velocidades actual
        w: parámetro de vorticidad
        h: paso para la diferencia finita
        
    Returns:
        gradiente: matriz del mismo tamaño que 'matriz'
    """
    nrows, ncols = matriz.shape
    gradiente = np.zeros_like(matriz)
    
    # Calculamos el residuo base
    residuo_base = F(matriz, w)
    
    # Para cada elemento, calculamos la derivada parcial
    for i in range(nrows):
        for j in range(ncols):
            # Si es un punto de la frontera izquierda (j=0), lo mantenemos fijo
            if j == 0:
                continue
                
            # Perturbamos el elemento (i,j)
            matriz_perturbada = matriz.copy()
            matriz_perturbada[i, j] += h
            
            # Calculamos el nuevo residuo
            residuo_perturbado = F(matriz_perturbada, w)
            
            # Aproximamos la derivada parcial usando diferencia finita
            delta_residuo = residuo_perturbado - residuo_base
            
            # La función objetivo es la suma de los cuadrados de los residuos
            # El gradiente es la derivada de esta función respecto a cada variable
            gradiente[i, j] = 2 * np.sum(residuo_base * delta_residuo) / h
            
    return gradiente

# Método del Gradiente Descendente
def gradiente_descendente(matriz_inicial, w=2, max_iter=1000, tolerancia=1e-10, tasa_aprendizaje=0.01):
    """
    Implementación del método de gradiente descendente para resolver el sistema.
    
    Args:
        matriz_inicial: matriz inicial de velocidades
        w: parámetro de vorticidad
        max_iter: número máximo de iteraciones
        tolerancia: criterio de parada
        tasa_aprendizaje: factor de ajuste del paso
        
    Returns:
        matriz: solución final
        historial_norma: historial de normas del residuo
    """
    matriz = matriz_inicial.copy()
    historial_norma = []
    
    # Fijamos los valores en la frontera izquierda (j=0)
    frontera_izquierda = matriz[:, 0].copy()
    
    for k in range(max_iter):
        # Calculamos el residuo actual
        residuo = F(matriz, w)
        
        # Calculamos la norma del residuo
        norma_residuo = np.linalg.norm(residuo)
        historial_norma.append(norma_residuo)
        
        # Verificamos convergencia
        if norma_residuo < tolerancia:
            print(f"Convergencia alcanzada en la iteración {k}, norma={norma_residuo}")
            break
        
        # Calculamos el gradiente
        gradiente = calcular_gradiente(matriz, w)
        
        # Actualizamos la matriz usando el gradiente descendente
        matriz = matriz - tasa_aprendizaje * gradiente
        
        # Restauramos los valores en la frontera
        matriz[:, 0] = frontera_izquierda
        
        # Monitoreo cada 100 iteraciones
        if k % 100 == 0:
            print(f"Iteración {k}, norma del residuo = {norma_residuo}")
            
        # Ajuste adaptativo de la tasa de aprendizaje
        if k > 0 and historial_norma[-1] > historial_norma[-2]:
            tasa_aprendizaje *= 0.7  # Reducimos la tasa si el error aumenta
        
    if k == max_iter - 1:
        print(f"Máximo de iteraciones alcanzado. Norma final = {norma_residuo}")
        
    return matriz, historial_norma

# Configuración principal
ny, nx = 7, 52

# Creamos la matriz inicial de velocidades
# Con un gradiente lineal de 1 a 0 en la dirección x
Vx_inicial = np.tile(np.linspace(1, 0, nx), (ny, 1))

# Resolvemos con Gradiente Descendente
print("Iniciando método de Gradiente Descendente...")
Vx_resultante, historial_norma = gradiente_descendente(
    Vx_inicial, 
    w=2, 
    max_iter=5000, 
    tolerancia=1e-8, 
    tasa_aprendizaje=0.005
)

# Graficar el resultado
plt.figure(figsize=(16, 8))

# Gráfico 1: La solución
plt.subplot(1, 2, 1)
plt.imshow(Vx_resultante, cmap='viridis', origin='lower', 
           extent=[0, 400, 0, 40], aspect='auto')
plt.colorbar(label='Velocidad Vx')
plt.title('Distribución de la velocidad Vx (Gradiente Descendente)')
plt.xlabel('Coordenada x (cm)')
plt.ylabel('Coordenada y (cm)')

# Gráfico 2: La convergencia
plt.subplot(1, 2, 2)
plt.semilogy(historial_norma)
plt.grid(True)
plt.title('Convergencia del método de Gradiente Descendente')
plt.xlabel('Iteración')
plt.ylabel('Norma del residuo (escala log)')

plt.tight_layout()
plt.show()