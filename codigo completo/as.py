import warnings
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import math

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

def limpiar_consola():
    """Limpiar la consola."""
    os.system('cls' if os.name == 'nt' else 'clear')

def imprimir_coloreado(texto, color, end='\n', flush=False):
    """Imprimir texto en color."""
    colores = {
        'rojo': '\033[91m',
        'verde': '\033[92m',
        'amarillo': '\033[93m',
        'azul': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'blanco': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colores.get(color, colores['reset'])}{texto}{colores['reset']}", end=end, flush=flush)

def animar_texto(texto, color='verde', delay=0.02):
    """Animar texto carácter por carácter en una línea horizontal."""
    for char in texto:
        imprimir_coloreado(char, color, end='', flush=True)
        time.sleep(delay)
    print()  # Nueva línea al final del texto

def animar_titulo(titulo):
    """Animar el título con un efecto de entrada."""
    for char in titulo:
        imprimir_coloreado(char, 'amarillo', end='', flush=True)
        time.sleep(0.05)
    print()  # Nueva línea al final del título

class OperacionesMatriz:
    """Clase abstracta para operaciones de matrices."""

    @staticmethod
    def validar_matrices(matriz1, matriz2=None):
        """Validar las dimensiones de las matrices según la operación."""
        if matriz2 is None:
            return matriz1.size > 0
        return matriz1.size > 0 and matriz2.size > 0

class SumaResta(OperacionesMatriz):
    """Clase para operaciones de suma y resta de matrices."""

    @staticmethod
    def validar_matrices(matriz1, matriz2):
        """Validar que las matrices tienen las mismas dimensiones."""
        if matriz1.shape != matriz2.shape:
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        return True

    @staticmethod
    def calcular_operacion(matriz1, matriz2, operacion):
        """Realizar la operación y mostrar el procedimiento paso a paso."""
        SumaResta.validar_matrices(matriz1, matriz2)

        operador = "+" if operacion == "suma" else "-"
        resultado = matriz1 + matriz2 if operacion == "suma" else matriz1 - matriz2

        imprimir_coloreado(f"\n{'='*50}\n{operacion.upper():^50}\n{'='*50}", 'magenta')

        max_len = max(len(str(x)) for x in np.concatenate([matriz1.flatten(), matriz2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz1.shape[0]):
            fila1 = " ".join(formato % str(x) for x in matriz1[i])
            fila2 = " ".join(formato % str(x) for x in matriz2[i])
            fila_res = " ".join(formato % str(x) for x in resultado[i])

            animar_texto(f"[ {fila1} ] {operador} [ {fila2} ] = [ {fila_res} ]", delay=0.01)
            time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Graficar el resultado con visualizaciones separadas
        UtilidadesMatriz.graficar_matriz_separada(matriz1, matriz2, resultado, operacion)

        # Graficar en 2D
        UtilidadesMatriz.graficar_2d(matriz1, matriz2, resultado, operacion)

        # Graficar en 3D si es posible
        if matriz1.shape[0] == 3 and matriz1.shape[1] == 3:
            UtilidadesMatriz.graficar_3d(matriz1, matriz2, resultado, operacion)

        return resultado

class Multiplicacion(OperacionesMatriz):
    """Clase para operaciones de multiplicación de matrices."""

    @staticmethod
    def validar_matrices(matriz1, matriz2):
        """Validar que las matrices son multiplicables."""
        if matriz1.shape[1] != matriz2.shape[0]:
            raise ValueError("El número de columnas de la primera matriz debe ser igual al número de filas de la segunda")
        return True

    @staticmethod
    def calcular_operacion(matriz1, matriz2):
        """Realizar la multiplicación y mostrar el procedimiento paso a paso."""
        Multiplicacion.validar_matrices(matriz1, matriz2)

        imprimir_coloreado(f"\n{'='*50}\n{'MULTIPLICACION DE MATRICES':^50}\n{'='*50}", 'magenta')

        # Convertir matrices a float64 para evitar problemas de conversión
        matriz1 = matriz1.astype(np.float64)
        matriz2 = matriz2.astype(np.float64)

        resultado = np.dot(matriz1, matriz2)
        filas1, columnas1 = matriz1.shape
        columnas2 = matriz2.shape[1]

        max_len = max(len(str(x)) for x in np.concatenate([matriz1.flatten(), matriz2.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(filas1):
            for j in range(columnas2):
                elementos = [f"({matriz1[i,k]} * {matriz2[k,j]})" for k in range(columnas1)]
                expresion = " + ".join(elementos)
                resultado_elem = resultado[i,j]

                imprimir_coloreado(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animar_texto(f"{expresion} = {resultado_elem:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Graficar el resultado con visualizaciones separadas
        UtilidadesMatriz.graficar_matriz_separada(matriz1, matriz2, resultado, "multiplicacion")

        # Graficar en 2D
        UtilidadesMatriz.graficar_2d(matriz1, matriz2, resultado, "multiplicacion")

        # Graficar en 3D si es posible
        if matriz1.shape[0] == 3 and matriz1.shape[1] == 3:
            UtilidadesMatriz.graficar_3d(matriz1, matriz2, resultado, "multiplicacion")

        return resultado

class EscalarVector(OperacionesMatriz):
    """Clase para operaciones de multiplicación escalar-vector."""

    @staticmethod
    def calcular_operacion(escalar, vector):
        """Realizar la multiplicación escalar-vector y mostrar el procedimiento paso a paso."""
        imprimir_coloreado(f"\n{'='*50}\n{'MULTIPLICACION ESCALAR-VECTOR':^50}\n{'='*50}", 'magenta')

        resultado = escalar * vector

        max_len = max(len(str(x)) for x in np.concatenate([vector.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector.shape[0]):
            imprimir_coloreado(f"\nElemento ({i+1}):", 'cyan')
            animar_texto(f"{escalar} * {vector[i]:>{max_len}} = {resultado[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Graficar en 2D
        UtilidadesMatriz.graficar_vector_2d(vector, resultado, "Multiplicación Escalar-Vector")

        return resultado

class PotenciaRaizLogaritmo(OperacionesMatriz):
    """Clase para operaciones de potencia, raíz y logaritmo de matrices."""

    @staticmethod
    def calcular_potencia(matriz, exponente):
        """Realizar la operación de potencia y mostrar el procedimiento paso a paso."""
        imprimir_coloreado(f"\n{'='*50}\n{'POTENCIA DE MATRIZ':^50}\n{'='*50}", 'magenta')

        resultado = np.power(matriz, exponente)

        max_len = max(len(str(x)) for x in np.concatenate([matriz.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                imprimir_coloreado(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animar_texto(f"{matriz[i,j]:>{max_len}} ^ {exponente} = {resultado[i,j]:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Graficar en 2D
        UtilidadesMatriz.graficar_2d(matriz, resultado, resultado, "Potencia de Matriz")

        return resultado

    @staticmethod
    def calcular_raiz(matriz, raiz):
        """Realizar la operación de raíz y mostrar el procedimiento paso a paso."""
        imprimir_coloreado(f"\n{'='*50}\n{'RAIZ DE MATRIZ':^50}\n{'='*50}", 'magenta')

        resultado = np.power(matriz, 1/raiz)

        max_len = max(len(str(x)) for x in np.concatenate([matriz.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                imprimir_coloreado(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animar_texto(f"{matriz[i,j]:>{max_len}} ^ (1/{raiz}) = {resultado[i,j]:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Graficar en 2D
        UtilidadesMatriz.graficar_2d(matriz, resultado, resultado, "Raíz de Matriz")

        return resultado

    @staticmethod
    def calcular_logaritmo(matriz, base=np.e):
        """Realizar la operación de logaritmo y mostrar el procedimiento paso a paso."""
        imprimir_coloreado(f"\n{'='*50}\n{'LOGARITMO DE MATRIZ':^50}\n{'='*50}", 'magenta')

        resultado = np.log(matriz) / np.log(base)

        max_len = max(len(str(x)) for x in np.concatenate([matriz.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                imprimir_coloreado(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animar_texto(f"log({matriz[i,j]:>{max_len}}, {base}) = {resultado[i,j]:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Graficar en 2D
        UtilidadesMatriz.graficar_2d(matriz, resultado, resultado, "Logaritmo de Matriz")

        return resultado

class Division(OperacionesMatriz):
    """Clase para operaciones de división de matrices."""

    @staticmethod
    def validar_matrices(matriz1, matriz2):
        """Validar que la segunda matriz es cuadrada e invertible."""
        if matriz2.shape[0] != matriz2.shape[1]:
            raise ValueError("La segunda matriz debe ser cuadrada")
        if abs(np.linalg.det(matriz2)) < 1e-10:
            raise ValueError("La segunda matriz no es invertible")
        return True

    @staticmethod
    def calcular_operacion(matriz1, matriz2):
        """Realizar la división y mostrar el procedimiento paso a paso."""
        Division.validar_matrices(matriz1, matriz2)

        imprimir_coloreado(f"\n{'='*50}\n{'DIVISION DE MATRICES':^50}\n{'='*50}", 'magenta')

        det = np.linalg.det(matriz2)
        imprimir_coloreado(f"\nDeterminante de la matriz: {det}", 'cyan')

        cofactores = np.zeros_like(matriz2, dtype=float)
        for i in range(matriz2.shape[0]):
            for j in range(matriz2.shape[1]):
                submatriz = np.delete(np.delete(matriz2, i, axis=0), j, axis=1)
                cofactores[i, j] = ((-1) ** (i + j)) * np.linalg.det(submatriz)

                imprimir_coloreado(f"\nCofactor ({i+1},{j+1}): {cofactores[i,j]:^10}", 'cyan')
                time.sleep(0.2)  # Pausa para efecto de movimiento

        adjunta = cofactores.T
        imprimir_coloreado("\nMatriz de Cofactores:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(cofactores, "Matriz de Cofactores")
        imprimir_coloreado("\nMatriz Adjunta (Transpuesta de la Matriz de Cofactores):", 'cyan')
        UtilidadesMatriz.imprimir_matriz(adjunta, "Matriz Adjunta")

        inversa = adjunta / det
        imprimir_coloreado("\nMatriz Inversa:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(inversa, "Matriz Inversa")

        resultado = np.dot(matriz1, inversa)
        imprimir_coloreado("\nProcedimiento de multiplicación:", 'cyan')
        filas1, columnas1 = matriz1.shape
        filas2 = inversa.shape[1]

        max_len = max(len(str(x)) for x in np.concatenate([matriz1.flatten(), inversa.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(filas1):
            for j in range(filas2):
                elementos = [f"({matriz1[i,k]} * {inversa[k,j]})" for k in range(columnas1)]
                expresion = " + ".join(elementos)
                resultado_elem = resultado[i,j]

                imprimir_coloreado(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animar_texto(f"{expresion} = {resultado_elem:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Graficar el resultado con visualizaciones separadas
        UtilidadesMatriz.graficar_matriz_separada(matriz1, matriz2, resultado, "division")

        # Graficar en 2D
        UtilidadesMatriz.graficar_2d(matriz1, matriz2, resultado, "division")

        # Graficar en 3D si es posible
        if matriz1.shape[0] == 3 and matriz1.shape[1] == 3:
            UtilidadesMatriz.graficar_3d(matriz1, matriz2, resultado, "division")

        return resultado

class Sarrus(OperacionesMatriz):
    """Clase para calcular el determinante de una matriz 3x3 usando la regla de Sarrus."""

    @staticmethod
    def validar_matriz(matriz):
        """Validar que la matriz es 3x3."""
        if matriz.shape != (3, 3):
            raise ValueError("La matriz debe ser de 3x3 para usar la regla de Sarrus")
        return True

    @staticmethod
    def calcular_determinante(matriz):
        """Calcular el determinante de una matriz 3x3 usando la regla de Sarrus."""
        Sarrus.validar_matriz(matriz)

        imprimir_coloreado(f"\n{'='*50}\n{'DETERMINANTE POR SARRUS':^50}\n{'='*50}", 'magenta')

        a, b, c = matriz[0]
        d, e, f = matriz[1]
        g, h, i = matriz[2]

        # Calcular el determinante usando la regla de Sarrus
        determinante = (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (a * f * h) - (b * d * i)

        # Imprimir las operaciones de Sarrus
        animar_texto(f"Determinante = ({a} * {e} * {i}) + ({b} * {f} * {g}) + ({c} * {d} * {h}) - ({c} * {e} * {g}) - ({a} * {f} * {h}) - ({b} * {d} * {i})", delay=0.01)
        imprimir_coloreado(f"Determinante = {determinante}", 'cyan')
        time.sleep(0.5)  # Pausa para efecto de movimiento

        # Calcular la matriz de cofactores
        cofactores = np.zeros_like(matriz, dtype=float)
        for i in range(3):
            for j in range(3):
                submatriz = np.delete(np.delete(matriz, i, axis=0), j, axis=1)
                cofactores[i, j] = ((-1) ** (i + j)) * np.linalg.det(submatriz)
                imprimir_coloreado(f"Cofactor ({i+1},{j+1}): (-1)^{i+j} * det(submatriz) = {cofactores[i,j]}", 'cyan')
                time.sleep(0.2)  # Pausa para efecto de movimiento

        # Calcular la matriz adjunta (transpuesta de la matriz de cofactores)
        adjunta = cofactores.T

        # Calcular la matriz inversa
        inversa = adjunta / determinante

        # Imprimir las matrices
        imprimir_coloreado("\nMatriz de Cofactores:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(cofactores, "Matriz de Cofactores")
        imprimir_coloreado("\nMatriz Adjunta (Transpuesta de la Matriz de Cofactores):", 'cyan')
        UtilidadesMatriz.imprimir_matriz(adjunta, "Matriz Adjunta")
        imprimir_coloreado("\nMatriz Inversa:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(inversa, "Matriz Inversa")

        # Graficar el determinante con visualizaciones separadas
        UtilidadesMatriz.graficar_determinante_separado(matriz, determinante)

        return determinante

class GaussJordan(OperacionesMatriz):
    """Clase para resolver sistemas de ecuaciones usando el método de Gauss-Jordan."""

    @staticmethod
    def validar_matriz(matriz):
        """Validar que la matriz es cuadrada."""
        if matriz.shape[0] != matriz.shape[1]:
            raise ValueError("La matriz debe ser cuadrada para usar Gauss-Jordan")
        return True

    @staticmethod
    def calcular_gauss_jordan(matriz):
        """Realizar la eliminación de Gauss-Jordan y mostrar el procedimiento paso a paso."""
        GaussJordan.validar_matriz(matriz)

        imprimir_coloreado(f"\n{'='*50}\n{'GAUSS-JORDAN':^50}\n{'='*50}", 'magenta')

        matriz_aumentada = np.hstack([matriz, np.identity(matriz.shape[0], dtype=np.float64)])
        n = matriz.shape[0]

        ecuaciones = []
        pasos_matrices = [matriz_aumentada.copy()]  # Almacenar cada paso de la matriz

        for i in range(n):
            # Hacer el pivote 1
            if matriz_aumentada[i, i] != 1:
                factor = 1 / matriz_aumentada[i, i]
                matriz_aumentada[i] *= factor
                ecuaciones.append(f"Fila {i+1}: Dividir cada elemento por {factor:.2f} para hacer el pivote 1")
                animar_texto(ecuaciones[-1], 'cyan')
                pasos_matrices.append(matriz_aumentada.copy())
                time.sleep(0.2)  # Pausa para efecto de movimiento

            # Hacer ceros debajo y encima del pivote
            for j in range(n):
                if i != j:
                    factor = matriz_aumentada[j, i]
                    matriz_aumentada[j] -= factor * matriz_aumentada[i]
                    ecuaciones.append(f"Fila {j+1}: Restar {factor:.2f} veces la fila {i+1} de la fila {j+1}")
                    animar_texto(ecuaciones[-1], 'cyan')
                    pasos_matrices.append(matriz_aumentada.copy())
                    time.sleep(0.2)  # Pausa para efecto de movimiento

        inversa = matriz_aumentada[:, n:]
        imprimir_coloreado("\nDesarrollo de las ecuaciones:", 'cyan')
        for eq in ecuaciones:
            animar_texto(eq, 'cyan')
            time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nMatriz Inversa:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(inversa, "Matriz Inversa")

        # Graficar el proceso de Gauss-Jordan con visualizaciones separadas
        UtilidadesMatriz.graficar_gauss_jordan_separado(matriz, inversa, pasos_matrices)

        return inversa

class OperacionesVector:
    """Clase para operaciones con vectores."""

    @staticmethod
    def validar_vectores(vector1, vector2=None):
        """Validar las dimensiones de los vectores según la operación."""
        if vector2 is None:
            return vector1.size > 0
        return vector1.size > 0 and vector2.size > 0 and vector1.shape == vector2.shape

    @staticmethod
    def suma_vectores(vector1, vector2):
        """Realizar la suma de vectores y mostrar el procedimiento paso a paso."""
        OperacionesVector.validar_vectores(vector1, vector2)

        imprimir_coloreado(f"\n{'='*50}\n{'SUMA DE VECTORES':^50}\n{'='*50}", 'magenta')

        resultado = vector1 + vector2

        max_len = max(len(str(x)) for x in np.concatenate([vector1.flatten(), vector2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector1.shape[0]):
            imprimir_coloreado(f"\nElemento ({i+1}):", 'cyan')
            animar_texto(f"{vector1[i]:>{max_len}} + {vector2[i]:>{max_len}} = {resultado[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Visualización si es posible (2D o 3D)
        dim = vector1.shape[0]
        if dim in [2, 3]:
            VisualizacionVector.visualizar_suma_vectores(vector1, vector2)

        return resultado

    @staticmethod
    def resta_vectores(vector1, vector2):
        """Realizar la resta de vectores y mostrar el procedimiento paso a paso."""
        OperacionesVector.validar_vectores(vector1, vector2)

        imprimir_coloreado(f"\n{'='*50}\n{'RESTA DE VECTORES':^50}\n{'='*50}", 'magenta')

        resultado = vector1 - vector2

        max_len = max(len(str(x)) for x in np.concatenate([vector1.flatten(), vector2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector1.shape[0]):
            imprimir_coloreado(f"\nElemento ({i+1}):", 'cyan')
            animar_texto(f"{vector1[i]:>{max_len}} - {vector2[i]:>{max_len}} = {resultado[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Visualización si es posible (2D o 3D)
        dim = vector1.shape[0]
        if dim in [2, 3]:
            VisualizacionVector.visualizar_resta_vectores(vector1, vector2)

        return resultado

    @staticmethod
    def producto_escalar(vector1, vector2):
        """Realizar el producto escalar (producto punto) y mostrar el procedimiento paso a paso."""
        OperacionesVector.validar_vectores(vector1, vector2)

        imprimir_coloreado(f"\n{'='*50}\n{'PRODUCTO ESCALAR':^50}\n{'='*50}", 'magenta')

        resultado = np.dot(vector1, vector2)

        max_len = max(len(str(x)) for x in np.concatenate([vector1.flatten(), vector2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector1.shape[0]):
            imprimir_coloreado(f"\nElemento ({i+1}):", 'cyan')
            animar_texto(f"{vector1[i]:>{max_len}} * {vector2[i]:>{max_len}} = {vector1[i] * vector2[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pausa para efecto de movimiento

        imprimir_coloreado("\nResultado:", 'cyan')
        imprimir_coloreado(f"Producto Escalar: {resultado}", 'cyan')

        # Visualización de los vectores
        dim = vector1.shape[0]
        if dim in [2, 3]:
            VisualizacionVector.visualizar_producto_escalar(vector1, vector2)

        return resultado

    @staticmethod
    def producto_vectorial(vector1, vector2):
        """Realizar el producto vectorial y mostrar el procedimiento paso a paso."""
        if vector1.shape[0] != 3 or vector2.shape[0] != 3:
            raise ValueError("Los vectores deben tener 3 componentes para el producto vectorial")

        imprimir_coloreado(f"\n{'='*50}\n{'PRODUCTO VECTORIAL':^50}\n{'='*50}", 'magenta')

        resultado = np.cross(vector1, vector2)

        imprimir_coloreado(f"Vector 1: {vector1}", 'cyan')
        imprimir_coloreado(f"Vector 2: {vector2}", 'cyan')
        imprimir_coloreado("\nResultado:", 'cyan')
        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado")

        # Visualización 3D
        VisualizacionVector.visualizar_producto_vectorial(vector1, vector2, resultado)

        return resultado

    @staticmethod
    def magnitud_vector(vector):
        """Calcular la magnitud de un vector y mostrar el procedimiento paso a paso."""
        OperacionesVector.validar_vectores(vector)

        imprimir_coloreado(f"\n{'='*50}\n{'MAGNITUD DEL VECTOR':^50}\n{'='*50}", 'magenta')

        resultado = np.linalg.norm(vector)

        imprimir_coloreado(f"Vector: {vector}", 'cyan')
        imprimir_coloreado("\nResultado:", 'cyan')
        imprimir_coloreado(f"Magnitud: {resultado}", 'cyan')

        # Visualización del vector y su magnitud
        dim = vector.shape[0]
        if dim in [2, 3]:
            VisualizacionVector.visualizar_magnitud_vector(vector)

        return resultado

    @staticmethod
    def angulo_entre_vectores(vector1, vector2):
        """Calcular el ángulo entre dos vectores y mostrar el procedimiento paso a paso."""
        OperacionesVector.validar_vectores(vector1, vector2)

        imprimir_coloreado(f"\n{'='*50}\n{'ANGULO ENTRE VECTORES':^50}\n{'='*50}", 'magenta')

        dot_product = np.dot(vector1, vector2)
        magnitud1 = np.linalg.norm(vector1)
        magnitud2 = np.linalg.norm(vector2)

        resultado = np.arccos(dot_product / (magnitud1 * magnitud2))
        resultado_deg = np.degrees(resultado)

        imprimir_coloreado(f"Vector 1: {vector1}", 'cyan')
        imprimir_coloreado(f"Vector 2: {vector2}", 'cyan')
        imprimir_coloreado("\nResultado:", 'cyan')
        imprimir_coloreado(f"Ángulo: {resultado_deg:.2f} grados", 'cyan')

        # Visualización si es posible (2D o 3D)
        dim = vector1.shape[0]
        if dim in [2, 3]:
            VisualizacionVector.visualizar_angulo_entre_vectores(vector1, vector2)

        return resultado_deg

class OperacionesCompletas(OperacionesMatriz):
    """Clase para realizar todas las operaciones de matrices en las mismas matrices."""

    @staticmethod
    def realizar_todas_operaciones(matriz1, matriz2):
        """Realizar todas las operaciones válidas en las matrices dadas."""
        imprimir_coloreado(f"\n{'='*50}\n{'ANÁLISIS COMPLETO DE MATRICES':^50}\n{'='*50}", 'magenta')

        operaciones_realizadas = []
        resultados = {}

        # Intentar realizar todas las operaciones
        try:
            # Suma
            imprimir_coloreado("\n1. SUMA DE MATRICES", 'amarillo')
            resultados['suma'] = SumaResta.calcular_operacion(matriz1, matriz2, "suma")
            operaciones_realizadas.append("suma")
        except ValueError as e:
            imprimir_coloreado(f"No se pudo realizar la suma: {e}", 'rojo')

        try:
            # Resta
            imprimir_coloreado("\n2. RESTA DE MATRICES", 'amarillo')
            resultados['resta'] = SumaResta.calcular_operacion(matriz1, matriz2, "resta")
            operaciones_realizadas.append("resta")
        except ValueError as e:
            imprimir_coloreado(f"No se pudo realizar la resta: {e}", 'rojo')

        try:
            # Multiplicación
            imprimir_coloreado("\n3. MULTIPLICACIÓN DE MATRICES", 'amarillo')
            resultados['multiplicacion'] = Multiplicacion.calcular_operacion(matriz1, matriz2)
            operaciones_realizadas.append("multiplicacion")
        except ValueError as e:
            imprimir_coloreado(f"No se pudo realizar la multiplicación: {e}", 'rojo')

        try:
            # División
            imprimir_coloreado("\n4. DIVISIÓN DE MATRICES", 'amarillo')
            resultados['division'] = Division.calcular_operacion(matriz1, matriz2)
            operaciones_realizadas.append("division")
        except ValueError as e:
            imprimir_coloreado(f"No se pudo realizar la división: {e}", 'rojo')

        # Determinantes (si son matrices cuadradas)
        if matriz1.shape[0] == matriz1.shape[1]:
            try:
                imprimir_coloreado(f"\n5. DETERMINANTE DE MATRIZ 1", 'amarillo')
                determinante1 = np.linalg.det(matriz1)
                imprimir_coloreado(f"Determinante: {determinante1}", 'cyan')
                resultados['determinante1'] = determinante1

                # Si es 3x3, usar Sarrus
                if matriz1.shape == (3, 3):
                    imprimir_coloreado("\nUsando Regla de Sarrus:", 'amarillo')
                    Sarrus.calcular_determinante(matriz1)

                operaciones_realizadas.append("determinante1")
            except Exception as e:
                imprimir_coloreado(f"No se pudo calcular el determinante de matriz 1: {e}", 'rojo')

            try:
                # Gauss-Jordan para matriz 1
                imprimir_coloreado(f"\n6. GAUSS-JORDAN PARA MATRIZ 1", 'amarillo')
                resultados['inversa1'] = GaussJordan.calcular_gauss_jordan(matriz1)
                operaciones_realizadas.append("gauss_jordan1")
            except Exception as e:
                imprimir_coloreado(f"No se pudo aplicar Gauss-Jordan a matriz 1: {e}", 'rojo')

        if matriz2.shape[0] == matriz2.shape[1]:
            try:
                imprimir_coloreado(f"\n7. DETERMINANTE DE MATRIZ 2", 'amarillo')
                determinante2 = np.linalg.det(matriz2)
                imprimir_coloreado(f"Determinante: {determinante2}", 'cyan')
                resultados['determinante2'] = determinante2

                # Si es 3x3, usar Sarrus
                if matriz2.shape == (3, 3):
                    imprimir_coloreado("\nUsando Regla de Sarrus:", 'amarillo')
                    Sarrus.calcular_determinante(matriz2)

                operaciones_realizadas.append("determinante2")
            except Exception as e:
                imprimir_coloreado(f"No se pudo calcular el determinante de matriz 2: {e}", 'rojo')

            try:
                # Gauss-Jordan para matriz 2
                imprimir_coloreado(f"\n8. GAUSS-JORDAN PARA MATRIZ 2", 'amarillo')
                resultados['inversa2'] = GaussJordan.calcular_gauss_jordan(matriz2)
                operaciones_realizadas.append("gauss_jordan2")
            except Exception as e:
                imprimir_coloreado(f"No se pudo aplicar Gauss-Jordan a matriz 2: {e}", 'rojo')

        # Resumen de todas las operaciones realizadas
        imprimir_coloreado(f"\n{'='*50}\n{'RESUMEN DE OPERACIONES':^50}\n{'='*50}", 'magenta')
        for op in operaciones_realizadas:
            if op == 'determinante1':
                imprimir_coloreado(f"- Determinante de Matriz 1: {resultados[op]}", 'verde')
            elif op == 'determinante2':
                imprimir_coloreado(f"- Determinante de Matriz 2: {resultados[op]}", 'verde')
            else:
                imprimir_coloreado(f"- {op.capitalize()} realizada correctamente", 'verde')

        return resultados

class UtilidadesMatriz:
    """Clase de utilidad para operaciones comunes con matrices."""

    @staticmethod
    def ingresar_matriz():
        """Permitir al usuario ingresar los elementos de una matriz de manera interactiva."""
        try:
            filas = int(input("Ingrese el número de filas: "))
            columnas = int(input("Ingrese el número de columnas: "))

            imprimir_coloreado(f"Ingrese los {filas * columnas} elementos separados por espacios (fila por fila):", 'cyan')
            elementos = []
            for i in range(filas):
                while True:
                    try:
                        fila = list(map(float, input(f"Fila {i+1}: ").split()))
                        if len(fila) != columnas:
                            raise ValueError(f"Número incorrecto de elementos en la fila {i+1}. Se esperaban {columnas} elementos.")
                        elementos.append(fila)
                        break
                    except ValueError as e:
                        imprimir_coloreado(f"Error: {e}", 'rojo')

            matriz = np.array(elementos)
            imprimir_coloreado("Matriz ingresada:", 'verde')
            UtilidadesMatriz.imprimir_matriz(matriz)
            confirmacion = input("¿Es correcta la matriz ingresada? (s/n): ").strip().lower()
            if confirmacion != 's':
                return UtilidadesMatriz.ingresar_matriz()

            return matriz

        except ValueError as e:
            imprimir_coloreado(f"Error al ingresar la matriz: {str(e)}", 'rojo')
            return np.array([])

    @staticmethod
    def generar_matriz_aleatoria():
        """Generar una matriz aleatoria con dimensiones definidas por el usuario."""
        try:
            filas = int(input("Ingrese el número de filas: "))
            columnas = int(input("Ingrese el número de columnas: "))

            min_val = float(input("Ingrese el valor mínimo para los elementos: "))
            max_val = float(input("Ingrese el valor máximo para los elementos: "))

            matriz = np.random.uniform(low=min_val, high=max_val, size=(filas, columnas))

            # Opción para redondear a enteros
            redondear = input("¿Desea redondear a números enteros? (s/n): ").strip().lower()
            if redondear == 's':
                matriz = np.round(matriz).astype(int)

            imprimir_coloreado("Matriz generada:", 'verde')
            UtilidadesMatriz.imprimir_matriz(matriz)

            return matriz

        except ValueError as e:
            imprimir_coloreado(f"Error al generar la matriz: {str(e)}", 'rojo')
            return np.array([])

    @staticmethod
    def ingresar_vector():
        """Permitir al usuario ingresar los elementos de un vector de manera interactiva."""
        try:
            dimension = int(input("Ingrese la dimensión del vector: "))

            imprimir_coloreado(f"Ingrese los {dimension} elementos separados por espacios:", 'cyan')
            elementos = list(map(float, input().split()))

            if len(elementos) != dimension:
                raise ValueError(f"Número incorrecto de elementos. Se esperaban {dimension} elementos.")

            vector = np.array(elementos)
            imprimir_coloreado("Vector ingresado:", 'verde')
            UtilidadesMatriz.imprimir_matriz(vector)
            confirmacion = input("¿Es correcto el vector ingresado? (s/n): ").strip().lower()
            if confirmacion != 's':
                return UtilidadesMatriz.ingresar_vector()

            return vector

        except ValueError as e:
            imprimir_coloreado(f"Error al ingresar el vector: {str(e)}", 'rojo')
            return np.array([])

    @staticmethod
    def generar_vector_aleatorio():
        """Generar un vector aleatorio con dimensión definida por el usuario."""
        try:
            dimension = int(input("Ingrese la dimensión del vector: "))

            min_val = float(input("Ingrese el valor mínimo para los elementos: "))
            max_val = float(input("Ingrese el valor máximo para los elementos: "))

            vector = np.random.uniform(low=min_val, high=max_val, size=dimension)

            # Opción para redondear a enteros
            redondear = input("¿Desea redondear a números enteros? (s/n): ").strip().lower()
            if redondear == 's':
                vector = np.round(vector).astype(int)

            imprimir_coloreado("Vector generado:", 'verde')
            UtilidadesMatriz.imprimir_matriz(vector)

            return vector

        except ValueError as e:
            imprimir_coloreado(f"Error al generar el vector: {str(e)}", 'rojo')
            return np.array([])

    @staticmethod
    def imprimir_matriz(matriz, titulo=""):
        """Imprimir una matriz con formato estructurado."""
        if matriz.size == 0:
            print("(Matriz vacía)")
            return

        max_len = max(len(str(x)) for x in matriz.flatten())
        formato = f"%{max_len}s"

        if titulo:
            imprimir_coloreado(f"\n{titulo:^50}", 'amarillo')

        print("-" * (matriz.shape[1] * (max_len + 2) + 2))
        for fila in matriz:
            print("| " + " ".join(formato % (str(int(x)) if x.is_integer() else str(x)) for x in fila) + " |")
        print("-" * (matriz.shape[1] * (max_len + 2) + 2))

    @staticmethod
    def imprimir_matrices_lado_a_lado(matriz1, matriz2, titulo=""):
        """Imprimir dos matrices lado a lado con formato estructurado."""
        if matriz1.size == 0 or matriz2.size == 0:
            print("(Matriz vacía)")
            return

        max_len1 = max(len(str(x)) for x in matriz1.flatten())
        max_len2 = max(len(str(x)) for x in matriz2.flatten())
        formato1 = f"%{max_len1}s"
        formato2 = f"%{max_len2}s"

        max_filas = max(matriz1.shape[0], matriz2.shape[0])

        imprimir_coloreado(f"\n{'='*50}\n{titulo:^50}\n{'='*50}", 'amarillo')
        imprimir_coloreado(f"{'Matriz 1':^{max_len1 * matriz1.shape[1] + 2}} | {'Matriz 2':^{max_len2 * matriz2.shape[1] + 2}}", 'cyan')
        print("-" * ((max_len1 + 2) * matriz1.shape[1] + 3 + (max_len2 + 2) * matriz2.shape[1]))

        for i in range(max_filas):
            fila1 = " ".join(formato1 % (str(int(x)) if x.is_integer() else str(x)) for x in matriz1[i]) if i < matriz1.shape[0] else " " * ((max_len1 + 2) * matriz1.shape[1])
            fila2 = " ".join(formato2 % (str(int(x)) if x.is_integer() else str(x)) for x in matriz2[i]) if i < matriz2.shape[0] else " " * ((max_len2 + 2) * matriz2.shape[1])
            imprimir_coloreado(f"| {fila1} | | {fila2} |", 'cyan')
            time.sleep(0.2)  # Pausa para efecto de movimiento

    @staticmethod
    def normalizar_datos_para_pie(data):
        """Normalizar datos para gráfico de pastel (hacer todos los valores positivos y asegurar que la suma > 0)."""
        abs_data = np.abs(data)
        if np.sum(abs_data) == 0:
            # Si todos los valores son cero, crear datos ficticios
            return np.ones_like(abs_data)
        return abs_data

    @staticmethod
    def crear_etiquetas(matriz, prefijo=''):
        """Crear etiquetas para los elementos de datos."""
        filas, cols = matriz.shape
        return [f"{prefijo}({i+1},{j+1}): {matriz[i,j]}" for i in range(filas) for j in range(cols)]

    @staticmethod
    def graficar_matriz_separada(matriz1, matriz2, resultado, operacion):
        """Graficar las matrices y el resultado como gráficos separados."""
        # Obtener datos aplanados
        matriz1_flat = matriz1.flatten()
        matriz2_flat = matriz2.flatten()
        resultado_flat = resultado.flatten()

        titulos = ['Matriz 1', 'Matriz 2', 'Resultado']
        conjuntos_datos = [matriz1_flat, matriz2_flat, resultado_flat]
        colores = ['skyblue', 'lightgreen', 'salmon']

        # Crear gráficos de barras (figuras separadas)
        for i, (titulo, datos, color) in enumerate(zip(titulos, conjuntos_datos, colores)):
            plt.figure(figsize=(10, 6))
            indices = range(len(datos))
            plt.bar(indices, datos, color=color, alpha=0.7)
            plt.title(f'{titulo}', fontsize=14)
            plt.xticks(indices, [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in indices] if len(datos) <= 20 else [])
            if len(datos) > 20:
                plt.xlabel(f"{len(datos)} elementos", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Crear gráficos de líneas (figuras separadas)
        for i, (titulo, datos, color) in enumerate(zip(titulos, conjuntos_datos, colores)):
            plt.figure(figsize=(10, 6))
            indices = range(len(datos))
            plt.plot(indices, datos, marker='o', linestyle='-', color=color)
            plt.title(f'{titulo}', fontsize=14)
            plt.xticks(indices, [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in indices] if len(datos) <= 20 else [])
            if len(datos) > 20:
                plt.xlabel(f"{len(datos)} elementos", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Crear histogramas (figuras separadas)
        for i, (titulo, datos, color) in enumerate(zip(titulos, conjuntos_datos, colores)):
            plt.figure(figsize=(10, 6))
            plt.hist(datos, bins=min(10, len(datos)), color=color, alpha=0.7, edgecolor='black')
            plt.title(f'{titulo}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Crear gráficos de dispersión (figuras separadas)
        for i, (titulo, datos, color) in enumerate(zip(titulos, conjuntos_datos, colores)):
            plt.figure(figsize=(10, 6))
            indices = range(len(datos))
            plt.scatter(indices, datos, color=color, alpha=0.7)
            plt.title(f'{titulo}', fontsize=14)
            plt.xticks(indices, [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in indices] if len(datos) <= 20 else [])
            if len(datos) > 20:
                plt.xlabel(f"{len(datos)} elementos", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Crear gráficos de pastel (figuras separadas)
        for i, (titulo, datos, color) in enumerate(zip(titulos, conjuntos_datos, colores)):
            plt.figure(figsize=(10, 6))
            pie_data = UtilidadesMatriz.normalizar_datos_para_pie(datos)
            if len(datos) <= 10:  # Solo mostrar etiquetas detalladas para matrices más pequeñas
                labels = [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in range(len(datos))]
                plt.pie(pie_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(datos))))
            else:
                plt.pie(pie_data, startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(datos))))
                plt.text(0, 0, f"{len(datos)} elementos", ha='center', va='center', fontsize=12)

            plt.title(f'{titulo}', fontsize=14)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    @staticmethod
    def graficar_determinante_separado(matriz, determinante):
        """Graficar la matriz y su determinante como gráficos separados."""
        plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=plt.gcf())

        # Obtener datos aplanados de la matriz
        matriz_flat = matriz.flatten()

        # Gráfico de barras
        ax1 = plt.subplot(gs[0, 0])
        indices = range(len(matriz_flat))
        ax1.bar(indices, matriz_flat, color='lightcoral', alpha=0.7)
        ax1.set_title('Matriz', fontsize=12)
        ax1.set_xticks(indices)
        ax1.set_xticklabels([f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in indices], fontsize=8, rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Gráfico de líneas
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(matriz_flat, marker='o', linestyle='-', color='lightcoral')
        ax2.set_title('Matriz', fontsize=12)
        ax2.set_xticks(indices)
        ax2.set_xticklabels([f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in indices], fontsize=8, rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Gráfico de dispersión
        ax3 = plt.subplot(gs[0, 2])
        ax3.scatter(indices, matriz_flat, color='lightcoral', alpha=0.7)
        ax3.set_title('Matriz', fontsize=12)
        ax3.set_xticks(indices)
        ax3.set_xticklabels([f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in indices], fontsize=8, rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.6)

        # Histograma
        ax4 = plt.subplot(gs[1, 0])
        ax4.hist(matriz_flat, bins=min(10, len(matriz_flat)), color='lightcoral', alpha=0.7, edgecolor='black')
        ax4.set_title('Matriz', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.6)

        # Gráfico de pastel para elementos de la matriz
        ax5 = plt.subplot(gs[1, 1])
        pie_data = UtilidadesMatriz.normalizar_datos_para_pie(matriz_flat)
        labels = [f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in range(len(matriz_flat))]
        ax5.pie(pie_data, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(matriz_flat))))
        ax5.set_title('Matriz', fontsize=12)

        # Visualización del determinante
        ax6 = plt.subplot(gs[1, 2])
        # Crear una representación visual del determinante
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.bar(['Determinante'], [determinante], color='purple' if determinante >= 0 else 'red', alpha=0.7)
        ax6.set_title(f'Determinante: {determinante}', fontsize=14)
        ax6.grid(True, linestyle='--', alpha=0.6)
        # Añadir anotación de texto mostrando el valor
        ax6.text(0, determinante/2, f"{determinante:.2f}", ha='center', fontsize=14,
                 color='white' if abs(determinante) > 3 else 'black')

        plt.suptitle(f'Análisis del Determinante de la Matriz', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def graficar_gauss_jordan_separado(matriz_original, inversa, pasos_matrices):
        """Graficar el proceso de eliminación de Gauss-Jordan como gráficos separados."""
        # Determinar cuántas visualizaciones mostrar
        num_pasos = min(4, len(pasos_matrices))  # Mostrar como máximo 4 pasos
        indices = np.linspace(0, len(pasos_matrices)-1, num_pasos, dtype=int)

        # Graficar matriz original
        matriz_flat = matriz_original.flatten()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(matriz_flat)), matriz_flat, color='royalblue', alpha=0.7)
        plt.title('Matriz Original', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Graficar matriz original como gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(matriz_flat)), matriz_flat, color='royalblue', alpha=0.7)
        plt.title('Matriz Original', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Graficar matriz original como histograma
        plt.figure(figsize=(10, 6))
        plt.hist(matriz_flat, bins=min(10, len(matriz_flat)), color='royalblue',
                 alpha=0.7, edgecolor='black')
        plt.title('Matriz Original', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Graficar pasos seleccionados del proceso de eliminación
        for i, idx in enumerate(indices):
            if i == len(indices) - 1:  # Saltar el último, mostraremos el resultado final por separado
                continue

            paso_matriz = pasos_matrices[idx]
            paso_matriz_flat = paso_matriz.flatten()

            # Gráfico de barras para este paso
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(paso_matriz_flat)), paso_matriz_flat,
                    color='lightgreen', alpha=0.7)
            plt.title(f'Paso {idx+1}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            # Gráfico de dispersión para este paso
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(paso_matriz_flat)), paso_matriz_flat,
                    color='lightgreen', alpha=0.7)
            plt.title(f'Paso {idx+1}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            # Histograma para este paso
            plt.figure(figsize=(10, 6))
            plt.hist(paso_matriz_flat, bins=min(10, len(paso_matriz_flat)),
                     color='lightgreen', alpha=0.7, edgecolor='black')
            plt.title(f'Paso {idx+1}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Graficar la matriz inversa final
        inversa_flat = inversa.flatten()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(inversa_flat)), inversa_flat, color='gold', alpha=0.7)
        plt.title('Matriz Inversa', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Gráfico de dispersión para la matriz inversa final
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(inversa_flat)), inversa_flat, color='gold', alpha=0.7)
        plt.title('Matriz Inversa', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Histograma para la matriz inversa final
        plt.figure(figsize=(10, 6))
        plt.hist(inversa_flat, bins=min(10, len(inversa_flat)),
                 color='gold', alpha=0.7, edgecolor='black')
        plt.title('Matriz Inversa', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def graficar_2d(matriz1, matriz2, resultado, operacion):
        """Graficar las matrices y el resultado en 2D."""
        plt.figure(figsize=(12, 6))

        # Graficar Matriz 1
        plt.subplot(1, 3, 1)
        plt.imshow(matriz1, cmap='viridis')
        plt.title('Matriz 1')
        plt.colorbar()

        # Graficar Matriz 2
        plt.subplot(1, 3, 2)
        plt.imshow(matriz2, cmap='viridis')
        plt.title('Matriz 2')
        plt.colorbar()

        # Graficar Resultado
        plt.subplot(1, 3, 3)
        plt.imshow(resultado, cmap='viridis')
        plt.title('Resultado')
        plt.colorbar()

        plt.suptitle(f'Operación: {operacion.capitalize()} (2D)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def graficar_3d(matriz1, matriz2, resultado, operacion):
        """Graficar las matrices y el resultado en 3D."""
        fig = plt.figure(figsize=(18, 6))

        # Graficar Matriz 1
        ax1 = fig.add_subplot(131, projection='3d')
        x, y = np.meshgrid(np.arange(matriz1.shape[1]), np.arange(matriz1.shape[0]))
        ax1.plot_surface(x, y, matriz1, cmap='viridis')
        ax1.set_title('Matriz 1')

        # Graficar Matriz 2
        ax2 = fig.add_subplot(132, projection='3d')
        x, y = np.meshgrid(np.arange(matriz2.shape[1]), np.arange(matriz2.shape[0]))
        ax2.plot_surface(x, y, matriz2, cmap='viridis')
        ax2.set_title('Matriz 2')

        # Graficar Resultado
        ax3 = fig.add_subplot(133, projection='3d')
        x, y = np.meshgrid(np.arange(resultado.shape[1]), np.arange(resultado.shape[0]))
        ax3.plot_surface(x, y, resultado, cmap='viridis')
        ax3.set_title('Resultado')

        plt.suptitle(f'Operación: {operacion.capitalize()} (3D)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def graficar_vector_2d(vector, resultado, operacion):
        """Graficar vectores y el resultado en 2D."""
        plt.figure(figsize=(12, 6))

        # Graficar Vector
        plt.subplot(1, 2, 1)
        plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector')
        plt.xlim(-1, max(vector[0], resultado[0]) + 1)
        plt.ylim(-1, max(vector[1], resultado[1]) + 1)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.title('Vector Original')
        plt.legend()

        # Graficar Resultado
        plt.subplot(1, 2, 2)
        plt.quiver(0, 0, resultado[0], resultado[1], angles='xy', scale_units='xy', scale=1, color='red', label='Resultado')
        plt.xlim(-1, max(vector[0], resultado[0]) + 1)
        plt.ylim(-1, max(vector[1], resultado[1]) + 1)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.title('Resultado')
        plt.legend()

        plt.suptitle(f'Operación: {operacion.capitalize()} (2D)')
        plt.tight_layout()
        plt.show()

class VisualizacionVector:
    """Clase para visualizar operaciones con vectores."""

    @staticmethod
    def visualizar_suma_vectores(vector1, vector2):
        """Visualizar la suma de vectores con animación."""
        resultado = vector1 + vector2

        fig, ax = plt.subplots()
        ax.set_xlim(-1, max(vector1[0], vector2[0], resultado[0]) + 1)
        ax.set_ylim(-1, max(vector1[1], vector2[1], resultado[1]) + 1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Inicializar quivers
        q1 = ax.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector 1')
        q2 = ax.quiver(vector1[0], vector1[1], vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color='green', label='Vector 2')
        q_result = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label='Resultado')

        def update(frame):
            if frame == 0:
                q_result.set_UVC(0, 0)
            elif frame == 1:
                q_result.set_UVC(vector1[0], vector1[1])
            elif frame == 2:
                q_result.set_UVC(resultado[0], resultado[1])
            return q1, q2, q_result

        ani = FuncAnimation(fig, update, frames=3, interval=1000, blit=False)
        plt.legend()
        plt.title('Suma de Vectores')
        plt.show()

    @staticmethod
    def visualizar_resta_vectores(vector1, vector2):
        """Visualizar la resta de vectores con animación."""
        resultado = vector1 - vector2

        fig, ax = plt.subplots()
        ax.set_xlim(-1, max(vector1[0], -vector2[0], resultado[0]) + 1)
        ax.set_ylim(-1, max(vector1[1], -vector2[1], resultado[1]) + 1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Inicializar quivers
        q1 = ax.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector 1')
        q2 = ax.quiver(vector1[0], vector1[1], -vector2[0], -vector2[1], angles='xy', scale_units='xy', scale=1, color='green', label='-Vector 2')
        q_result = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label='Resultado')

        def update(frame):
            if frame == 0:
                q_result.set_UVC(0, 0)
            elif frame == 1:
                q_result.set_UVC(vector1[0], vector1[1])
            elif frame == 2:
                q_result.set_UVC(resultado[0], resultado[1])
            return q1, q2, q_result

        ani = FuncAnimation(fig, update, frames=3, interval=1000, blit=False)
        plt.legend()
        plt.title('Resta de Vectores')
        plt.show()

    @staticmethod
    def visualizar_producto_vectorial(vector1, vector2, resultado):
        """Visualizar el producto vectorial."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, vector1[0], vector1[1], vector1[2], color='blue', label='Vector 1')
        ax.quiver(0, 0, 0, vector2[0], vector2[1], vector2[2], color='green', label='Vector 2')
        ax.quiver(0, 0, 0, resultado[0], resultado[1], resultado[2], color='red', label='Resultado')
        ax.set_xlim([-1, max(vector1[0], vector2[0], resultado[0]) + 1])
        ax.set_ylim([-1, max(vector1[1], vector2[1], resultado[1]) + 1])
        ax.set_zlim([-1, max(vector1[2], vector2[2], resultado[2]) + 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Producto Vectorial')
        plt.show()

    @staticmethod
    def visualizar_angulo_entre_vectores(vector1, vector2):
        """Visualizar el ángulo entre vectores."""
        plt.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector 1')
        plt.quiver(0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color='green', label='Vector 2')
        plt.xlim(-1, max(vector1[0], vector2[0]) + 1)
        plt.ylim(-1, max(vector1[1], vector2[1]) + 1)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.title('Ángulo entre Vectores')
        plt.show()

    @staticmethod
    def visualizar_producto_escalar(escalar, vector):
        """Visualizar la multiplicación escalar con animación."""
        resultado = escalar * vector

        fig, ax = plt.subplots()
        ax.set_xlim(-1, max(vector[0], resultado[0]) + 1)
        ax.set_ylim(-1, max(vector[1], resultado[1]) + 1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Inicializar quivers
        q1 = ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector Original')
        q_result = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label='Vector Escalado')

        def update(frame):
            if frame == 0:
                q_result.set_UVC(0, 0)
            elif frame == 1:
                q_result.set_UVC(resultado[0], resultado[1])
            return q1, q_result

        ani = FuncAnimation(fig, update, frames=2, interval=1000, blit=False)
        plt.legend()
        plt.title('Producto Escalar por Vector')
        plt.show()

    @staticmethod
    def visualizar_proyeccion(vector1, vector2, resultado):
        """Visualizar la proyección de vector1 sobre vector2 con animación."""
        fig, ax = plt.subplots()
        ax.set_xlim(-1, max(vector1[0], vector2[0], resultado[0]) + 1)
        ax.set_ylim(-1, max(vector1[1], vector2[1], resultado[1]) + 1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Inicializar quivers
        q1 = ax.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector 1')
        q2 = ax.quiver(0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color='green', label='Vector 2')
        q_result = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label='Proyección')

        def update(frame):
            if frame == 0:
                q_result.set_UVC(0, 0)
            elif frame == 1:
                q_result.set_UVC(resultado[0], resultado[1])
            return q1, q2, q_result

        ani = FuncAnimation(fig, update, frames=2, interval=1000, blit=False)
        plt.legend()
        plt.title('Proyección de Vector')
        plt.show()

    @staticmethod
    def visualizar_magnitud_vector(vector):
        """Visualizar la magnitud de un vector."""
        fig, ax = plt.subplots()
        ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector')
        ax.set_xlim(-1, vector[0] + 1)
        ax.set_ylim(-1, vector[1] + 1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.title('Magnitud del Vector')
        plt.show()

def main():
    limpiar_consola()
    animar_titulo("Calculadora de Matrices y Vectores")

    while True:
        tipo_operacion = input("¿Desea trabajar con matrices o vectores? (m/v): ").strip().lower()

        if tipo_operacion == 'm':
            imprimir_coloreado("\nIngrese el número de filas y columnas de la primera matriz:", 'cyan')
            filas1 = int(input("Ingrese el número de filas: "))
            columnas1 = int(input("Ingrese el número de columnas: "))

            imprimir_coloreado(f"Ingrese los {filas1 * columnas1} elementos separados por espacios (fila por fila):", 'cyan')
            elementos1 = []
            for i in range(filas1):
                while True:
                    try:
                        fila = list(map(float, input(f"Fila {i+1}: ").split()))
                        if len(fila) != columnas1:
                            raise ValueError(f"Número incorrecto de elementos en la fila {i+1}. Se esperaban {columnas1} elementos.")
                        elementos1.append(fila)
                        break
                    except ValueError as e:
                        imprimir_coloreado(f"Error: {e}", 'rojo')

            matriz1 = np.array(elementos1)
            imprimir_coloreado("Primera matriz ingresada:", 'verde')
            UtilidadesMatriz.imprimir_matriz(matriz1)

            imprimir_coloreado("\nIngrese el número de filas y columnas de la segunda matriz:", 'cyan')
            filas2 = int(input("Ingrese el número de filas: "))
            columnas2 = int(input("Ingrese el número de columnas: "))

            imprimir_coloreado(f"Ingrese los {filas2 * columnas2} elementos separados por espacios (fila por fila):", 'cyan')
            elementos2 = []
            for i in range(filas2):
                while True:
                    try:
                        fila = list(map(float, input(f"Fila {i+1}: ").split()))
                        if len(fila) != columnas2:
                            raise ValueError(f"Número incorrecto de elementos en la fila {i+1}. Se esperaban {columnas2} elementos.")
                        elementos2.append(fila)
                        break
                    except ValueError as e:
                        imprimir_coloreado(f"Error: {e}", 'rojo')

            matriz2 = np.array(elementos2)
            imprimir_coloreado("Segunda matriz ingresada:", 'verde')
            UtilidadesMatriz.imprimir_matriz(matriz2)

            imprimir_coloreado("\nSeleccione una operación:", 'cyan')
            imprimir_coloreado("1. Suma de matrices", 'amarillo')
            imprimir_coloreado("2. Resta de matrices", 'amarillo')
            imprimir_coloreado("3. Multiplicación de matrices", 'amarillo')
            imprimir_coloreado("4. División de matrices", 'amarillo')
            imprimir_coloreado("5. Determinante por Sarrus (3x3)", 'amarillo')
            imprimir_coloreado("6. Gauss-Jordan", 'amarillo')
            imprimir_coloreado("7. Multiplicación escalar-vector", 'amarillo')
            imprimir_coloreado("8. Potencia de matriz", 'amarillo')
            imprimir_coloreado("9. Raíz de matriz", 'amarillo')
            imprimir_coloreado("10. Logaritmo de matriz", 'amarillo')
            imprimir_coloreado("11. Multiplicación matriz A1X1", 'amarillo')
            imprimir_coloreado("12. Análisis completo de matrices", 'amarillo')
            imprimir_coloreado("13. Salir", 'amarillo')

            opcion = input("Ingrese el número de la operación deseada: ").strip()

            if opcion == '1':
                resultado = SumaResta.calcular_operacion(matriz1, matriz2, "suma")
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Suma")

            elif opcion == '2':
                resultado = SumaResta.calcular_operacion(matriz1, matriz2, "resta")
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Resta")

            elif opcion == '3':
                resultado = Multiplicacion.calcular_operacion(matriz1, matriz2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Multiplicación")

            elif opcion == '4':
                resultado = Division.calcular_operacion(matriz1, matriz2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la División")

            elif opcion == '5':
                if matriz1.shape == (3, 3):
                    resultado = Sarrus.calcular_determinante(matriz1)
                    imprimir_coloreado(f"Determinante de la matriz 1: {resultado}", 'cyan')
                else:
                    imprimir_coloreado("La matriz 1 no es 3x3, no se puede aplicar la regla de Sarrus.", 'rojo')

                if matriz2.shape == (3, 3):
                    resultado = Sarrus.calcular_determinante(matriz2)
                    imprimir_coloreado(f"Determinante de la matriz 2: {resultado}", 'cyan')
                else:
                    imprimir_coloreado("La matriz 2 no es 3x3, no se puede aplicar la regla de Sarrus.", 'rojo')

            elif opcion == '6':
                if matriz1.shape[0] == matriz1.shape[1]:
                    resultado = GaussJordan.calcular_gauss_jordan(matriz1)
                    UtilidadesMatriz.imprimir_matriz(resultado, "Matriz Inversa de la Matriz 1")
                else:
                    imprimir_coloreado("La matriz 1 no es cuadrada, no se puede aplicar Gauss-Jordan.", 'rojo')

                if matriz2.shape[0] == matriz2.shape[1]:
                    resultado = GaussJordan.calcular_gauss_jordan(matriz2)
                    UtilidadesMatriz.imprimir_matriz(resultado, "Matriz Inversa de la Matriz 2")
                else:
                    imprimir_coloreado("La matriz 2 no es cuadrada, no se puede aplicar Gauss-Jordan.", 'rojo')

            elif opcion == '7':
                escalar = float(input("Ingrese el valor del escalar: "))
                vector = np.array(list(map(float, input("Ingrese los elementos del vector separados por espacios: ").split())))
                resultado = EscalarVector.calcular_operacion(escalar, vector)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Multiplicación Escalar-Vector")

            elif opcion == '8':
                exponente = float(input("Ingrese el valor del exponente: "))
                resultado1 = PotenciaRaizLogaritmo.calcular_potencia(matriz1, exponente)
                UtilidadesMatriz.imprimir_matriz(resultado1, "Resultado de la Potencia de la Matriz 1")
                resultado2 = PotenciaRaizLogaritmo.calcular_potencia(matriz2, exponente)
                UtilidadesMatriz.imprimir_matriz(resultado2, "Resultado de la Potencia de la Matriz 2")

            elif opcion == '9':
                raiz = float(input("Ingrese el valor de la raíz: "))
                resultado1 = PotenciaRaizLogaritmo.calcular_raiz(matriz1, raiz)
                UtilidadesMatriz.imprimir_matriz(resultado1, "Resultado de la Raíz de la Matriz 1")
                resultado2 = PotenciaRaizLogaritmo.calcular_raiz(matriz2, raiz)
                UtilidadesMatriz.imprimir_matriz(resultado2, "Resultado de la Raíz de la Matriz 2")

            elif opcion == '10':
                base = float(input("Ingrese el valor de la base del logaritmo: "))
                resultado1 = PotenciaRaizLogaritmo.calcular_logaritmo(matriz1, base)
                UtilidadesMatriz.imprimir_matriz(resultado1, "Resultado del Logaritmo de la Matriz 1")
                resultado2 = PotenciaRaizLogaritmo.calcular_logaritmo(matriz2, base)
                UtilidadesMatriz.imprimir_matriz(resultado2, "Resultado del Logaritmo de la Matriz 2")

            elif opcion == '11':
                if matriz1.shape[1] == 1:
                    vector = np.array(list(map(float, input("Ingrese los elementos del vector separados por espacios: ").split())))
                    if vector.shape[0] == matriz1.shape[0]:
                        resultado = np.dot(matriz1, vector)
                        UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Multiplicación A1X1")
                    else:
                        imprimir_coloreado("El vector no tiene la misma cantidad de filas que la matriz.", 'rojo')
                else:
                    imprimir_coloreado("La matriz 1 no tiene una sola columna.", 'rojo')

            elif opcion == '12':
                resultados = OperacionesCompletas.realizar_todas_operaciones(matriz1, matriz2)

            elif opcion == '13':
                imprimir_coloreado("Saliendo del programa...", 'rojo')
                break

            else:
                imprimir_coloreado("Opción no válida. Por favor, intente de nuevo.", 'rojo')

        elif tipo_operacion == 'v':
            imprimir_coloreado("\nIngrese la dimensión del primer vector:", 'cyan')
            dimension1 = int(input("Ingrese la dimensión: "))

            imprimir_coloreado(f"Ingrese los {dimension1} elementos separados por espacios:", 'cyan')
            vector1 = UtilidadesMatriz.ingresar_vector()

            imprimir_coloreado("\nIngrese la dimensión del segundo vector:", 'cyan')
            dimension2 = int(input("Ingrese la dimensión: "))

            imprimir_coloreado(f"Ingrese los {dimension2} elementos separados por espacios:", 'cyan')
            vector2 = UtilidadesMatriz.ingresar_vector()

            imprimir_coloreado("\nSeleccione una operación:", 'cyan')
            imprimir_coloreado("1. Suma de vectores", 'amarillo')
            imprimir_coloreado("2. Resta de vectores", 'amarillo')
            imprimir_coloreado("3. Producto escalar", 'amarillo')
            imprimir_coloreado("4. Producto vectorial", 'amarillo')
            imprimir_coloreado("5. Magnitud del vector", 'amarillo')
            imprimir_coloreado("6. Ángulo entre vectores", 'amarillo')
            imprimir_coloreado("7. Producto escalar por vector", 'amarillo')
            imprimir_coloreado("8. Proyección de vector", 'amarillo')
            imprimir_coloreado("9. Componentes rectangulares", 'amarillo')
            imprimir_coloreado("10. Dirección en puntos cardinales", 'amarillo')
            imprimir_coloreado("11. Distancia recorrida", 'amarillo')
            imprimir_coloreado("12. Desplazamiento", 'amarillo')
            imprimir_coloreado("13. Transformación lineal", 'amarillo')
            imprimir_coloreado("14. Rotación 2D", 'amarillo')
            imprimir_coloreado("15. Multiplicación de vectores", 'amarillo')
            imprimir_coloreado("16. División de vectores", 'amarillo')
            imprimir_coloreado("17. Potencia de vector", 'amarillo')
            imprimir_coloreado("18. Raíz logarítmica de vector", 'amarillo')
            imprimir_coloreado("19. Normalización de vector", 'amarillo')
            imprimir_coloreado("20. Reflexión de vector", 'amarillo')
            imprimir_coloreado("21. Suma de múltiples vectores", 'amarillo')
            imprimir_coloreado("22. Descomposición de vector", 'amarillo')
            imprimir_coloreado("23. Área del paralelogramo", 'amarillo')
            imprimir_coloreado("24. Independencia lineal", 'amarillo')
            imprimir_coloreado("25. Transformación cero", 'amarillo')
            imprimir_coloreado("26. Transformación identidad", 'amarillo')
            imprimir_coloreado("27. Transformación de reflexión", 'amarillo')
            imprimir_coloreado("28. Transformación de rotación", 'amarillo')
            imprimir_coloreado("29. Transformación de escala", 'amarillo')
            imprimir_coloreado("30. Transformación de traslación", 'amarillo')
            imprimir_coloreado("31. Autovalores y autovectores", 'amarillo')
            imprimir_coloreado("32. Salir", 'amarillo')

            opcion = input("Ingrese el número de la operación deseada: ").strip()

            if opcion == '1':
                resultado = OperacionesVector.suma_vectores(vector1, vector2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Suma de Vectores")

            elif opcion == '2':
                resultado = OperacionesVector.resta_vectores(vector1, vector2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Resta de Vectores")

            elif opcion == '3':
                resultado = OperacionesVector.producto_escalar(vector1, vector2)
                imprimir_coloreado(f"Producto Escalar: {resultado}", 'cyan')

            elif opcion == '4':
                resultado = OperacionesVector.producto_vectorial(vector1, vector2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado del Producto Vectorial")

            elif opcion == '5':
                resultado1 = OperacionesVector.magnitud_vector(vector1)
                imprimir_coloreado(f"Magnitud del Vector 1: {resultado1}", 'cyan')
                resultado2 = OperacionesVector.magnitud_vector(vector2)
                imprimir_coloreado(f"Magnitud del Vector 2: {resultado2}", 'cyan')

            elif opcion == '6':
                resultado = OperacionesVector.angulo_entre_vectores(vector1, vector2)
                imprimir_coloreado(f"Ángulo entre Vectores: {resultado} grados", 'cyan')

            elif opcion == '7':
                escalar = float(input("Ingrese el valor del escalar: "))
                resultado = OperacionesVector.producto_escalar_por_vector(escalar, vector1)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado del Producto Escalar por Vector")

            elif opcion == '8':
                resultado = OperacionesVector.proyeccion_vector(vector1, vector2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Proyección de Vector")

            elif opcion == '9':
                resultado = OperacionesVector.componentes_rectangulares(vector1)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de Componentes Rectangulares")

            elif opcion == '10':
                resultado = OperacionesVector.direccion_puntos_cardinales(vector1)
                imprimir_coloreado(f"Dirección en Puntos Cardinales: {resultado}", 'cyan')

            elif opcion == '11':
                puntos = np.array([list(map(float, input(f"Ingrese las componentes del punto {i+1} separadas por espacio: ").split())) for i in range(int(input("Ingrese el número de puntos: ")))])
                resultado = OperacionesVector.distancia_recorrida(puntos)
                imprimir_coloreado(f"Distancia Recorrida: {resultado}", 'cyan')

            elif opcion == '12':
                punto_inicial = np.array(list(map(float, input("Ingrese las componentes del punto inicial separadas por espacio: ").split())))
                punto_final = np.array(list(map(float, input("Ingrese las componentes del punto final separadas por espacio: ").split())))
                resultado = OperacionesVector.desplazamiento(punto_inicial, punto_final)
                imprimir_coloreado(f"Desplazamiento: {resultado}", 'cyan')

            elif opcion == '13':
                filas = int(input("Ingrese el número de filas de la matriz de transformación: "))
                columnas = int(input("Ingrese el número de columnas de la matriz de transformación: "))
                matriz_transformacion = np.array([list(map(float, input(f"Ingrese los elementos de la fila {i+1} separadas por espacio: ").split())) for i in range(filas)])
                resultado = OperacionesVector.transformacion_lineal(vector1, matriz_transformacion)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Transformación Lineal")

            elif opcion == '14':
                angulo_grados = float(input("Ingrese el ángulo de rotación en grados: "))
                resultado = OperacionesVector.rotacion_2d(vector1, angulo_grados)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Rotación 2D")

            elif opcion == '15':
                resultado = OperacionesVector.multiplicar_vectores(vector1, vector2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Multiplicación de Vectores")

            elif opcion == '16':
                resultado = OperacionesVector.dividir_vectores(vector1, vector2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la División de Vectores")

            elif opcion == '17':
                exponente = float(input("Ingrese el exponente: "))
                resultado = OperacionesVector.potencia_vector(vector1, exponente)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Potencia de Vector")

            elif opcion == '18':
                resultado = OperacionesVector.raiz_logaritmo_vector(vector1)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Raíz Logarítmica de Vector")

            elif opcion == '19':
                resultado = OperacionesVector.normalizar_vector(vector1)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Normalización de Vector")

            elif opcion == '20':
                resultado = OperacionesVector.reflexion_vector(vector1, vector2)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Reflexión de Vector")

            elif opcion == '21':
                vectores = [np.array(list(map(float, input(f"Ingrese las componentes del vector {i+1} separadas por espacio: ").split()))) for i in range(int(input("Ingrese el número de vectores: ")))]
                resultado = OperacionesVector.sumar_multiples_vectores(vectores)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Suma de Múltiples Vectores")

            elif opcion == '22':
                resultado = OperacionesVector.descomponer_vector(vector1, vector2)
                imprimir_coloreado(f"Descomposición de Vector: {resultado}", 'cyan')

            elif opcion == '23':
                resultado = OperacionesVector.area_paralelogramo(vector1, vector2)
                imprimir_coloreado(f"Área del Paralelogramo: {resultado}", 'cyan')

            elif opcion == '24':
                resultado = OperacionesVector.son_linealmente_independientes(vector1, vector2)
                imprimir_coloreado(f"Independencia Lineal: {resultado}", 'cyan')

            elif opcion == '25':
                resultado = OperacionesVector.transformacion_cero(vector1)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Transformación Cero")

            elif opcion == '26':
                resultado = OperacionesVector.transformacion_identidad(vector1)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Transformación Identidad")

            elif opcion == '27':
                eje = input("Ingrese el eje de reflexión (x o y): ")
                resultado = OperacionesVector.transformacion_reflexion(vector1, eje)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Transformación de Reflexión")

            elif opcion == '28':
                angulo_grados = float(input("Ingrese el ángulo de rotación en grados: "))
                resultado = OperacionesVector.transformacion_rotacion(vector1, angulo_grados)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Transformación de Rotación")

            elif opcion == '29':
                factor = float(input("Ingrese el factor de escala: "))
                resultado = OperacionesVector.transformacion_escala(vector1, factor)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Transformación de Escala")

            elif opcion == '30':
                traslacion = np.array(list(map(float, input("Ingrese las componentes del vector de traslación separadas por espacio: ").split())))
                resultado = OperacionesVector.transformacion_traslacion(vector1, traslacion)
                UtilidadesMatriz.imprimir_matriz(resultado, "Resultado de la Transformación de Traslación")

            elif opcion == '31':
                filas = int(input("Ingrese el número de filas de la matriz: "))
                columnas = int(input("Ingrese el número de columnas de la matriz: "))
                matriz = np.array([list(map(float, input(f"Ingrese los elementos de la fila {i+1} separadas por espacio: ").split())) for i in range(filas)])
                resultado = OperacionesVector.autovalores_y_autovectores(matriz)
                imprimir_coloreado(f"Autovalores y Autovectores: {resultado}", 'cyan')

            elif opcion == '32':
                imprimir_coloreado("Saliendo del programa...", 'rojo')
                break

            else:
                imprimir_coloreado("Opción no válida. Por favor, intente de nuevo.", 'rojo')

        else:
            imprimir_coloreado("Opción no válida. Por favor, intente de nuevo.", 'rojo')

if __name__ == "__main__":
    main()
