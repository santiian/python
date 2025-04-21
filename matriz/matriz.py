import warnings
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

def clear_console():
    """Clear the console."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored(text, color, end='\n', flush=False):
    """Print text in color."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}", end=end, flush=flush)

def animate_text(text, color='green', delay=0.02):
    """Animate text character by character in a horizontal line."""
    for char in text:
        print_colored(char, color, end='', flush=True)
        time.sleep(delay)
    print()  # New line at the end of the text

def animate_title(title):
    """Animate the title with an entrance effect."""
    for char in title:
        print_colored(char, 'yellow', end='', flush=True)
        time.sleep(0.05)
    print()  # New line at the end of the title

class MatrizOperaciones:
    """Abstract class for matrix operations."""

    @staticmethod
    def validar_matrices(matriz1, matriz2=None):
        """Validate the dimensions of the matrices according to the operation."""
        if matriz2 is None:
            return matriz1.size > 0
        return matriz1.size > 0 and matriz2.size > 0

class SumaResta(MatrizOperaciones):
    """Class for addition and subtraction operations of matrices."""

    @staticmethod
    def validar_matrices(matriz1, matriz2):
        """Validate that the matrices have the same dimensions."""
        if matriz1.shape != matriz2.shape:
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        return True

    @staticmethod
    def calcular_operacion(matriz1, matriz2, operacion):
        """Perform the operation and show the step-by-step procedure."""
        SumaResta.validar_matrices(matriz1, matriz2)

        operador = "+" if operacion == "suma" else "-"
        resultado = matriz1 + matriz2 if operacion == "suma" else matriz1 - matriz2

        print_colored(f"\n{'='*50}\n{operacion.upper():^50}\n{'='*50}", 'magenta')

        max_len = max(len(str(x)) for x in np.concatenate([matriz1.flatten(), matriz2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz1.shape[0]):
            fila1 = " ".join(formato % str(x) for x in matriz1[i])
            fila2 = " ".join(formato % str(x) for x in matriz2[i])
            fila_res = " ".join(formato % str(x) for x in resultado[i])

            animate_text(f"[ {fila1} ] {operador} [ {fila2} ] = [ {fila_res} ]", delay=0.01)
            time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        # Plot the result with separate visualizations
        MatrizUtil.plot_matriz_separate(matriz1, matriz2, resultado, operacion)

        return resultado

class Multiplicacion(MatrizOperaciones):
    """Class for matrix multiplication operations."""

    @staticmethod
    def validar_matrices(matriz1, matriz2):
        """Validate that the matrices are multipliable."""
        if matriz1.shape[1] != matriz2.shape[0]:
            raise ValueError("El número de columnas de la primera matriz debe ser igual al número de filas de la segunda")
        return True

    @staticmethod
    def calcular_operacion(matriz1, matriz2):
        """Perform the multiplication and show the step-by-step procedure."""
        Multiplicacion.validar_matrices(matriz1, matriz2)

        print_colored(f"\n{'='*50}\n{'MULTIPLICACION DE MATRICES':^50}\n{'='*50}", 'magenta')

        # Convert matrices to float64 to avoid casting issues
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

                print_colored(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animate_text(f"{expresion} = {resultado_elem:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        # Plot the result with separate visualizations
        MatrizUtil.plot_matriz_separate(matriz1, matriz2, resultado, "multiplicacion")

        return resultado

class EscalarVector(MatrizOperaciones):
    """Class for scalar-vector multiplication operations."""

    @staticmethod
    def calcular_operacion(escalar, vector):
        """Perform the scalar-vector multiplication and show the step-by-step procedure."""
        print_colored(f"\n{'='*50}\n{'MULTIPLICACION ESCALAR-VECTOR':^50}\n{'='*50}", 'magenta')

        resultado = escalar * vector

        max_len = max(len(str(x)) for x in np.concatenate([vector.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector.shape[0]):
            print_colored(f"\nElemento ({i+1}):", 'cyan')
            animate_text(f"{escalar} * {vector[i]:>{max_len}} = {resultado[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        return resultado

class PotenciaRaizLogaritmo(MatrizOperaciones):
    """Class for power, root, and logarithm operations on matrices."""

    @staticmethod
    def calcular_potencia(matriz, exponente):
        """Perform the power operation and show the step-by-step procedure."""
        print_colored(f"\n{'='*50}\n{'POTENCIA DE MATRIZ':^50}\n{'='*50}", 'magenta')

        resultado = np.power(matriz, exponente)

        max_len = max(len(str(x)) for x in np.concatenate([matriz.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                print_colored(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animate_text(f"{matriz[i,j]:>{max_len}} ^ {exponente} = {resultado[i,j]:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        return resultado

    @staticmethod
    def calcular_raiz(matriz, raiz):
        """Perform the root operation and show the step-by-step procedure."""
        print_colored(f"\n{'='*50}\n{'RAIZ DE MATRIZ':^50}\n{'='*50}", 'magenta')

        resultado = np.power(matriz, 1/raiz)

        max_len = max(len(str(x)) for x in np.concatenate([matriz.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                print_colored(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animate_text(f"{matriz[i,j]:>{max_len}} ^ (1/{raiz}) = {resultado[i,j]:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        return resultado

    @staticmethod
    def calcular_logaritmo(matriz, base=np.e):
        """Perform the logarithm operation and show the step-by-step procedure."""
        print_colored(f"\n{'='*50}\n{'LOGARITMO DE MATRIZ':^50}\n{'='*50}", 'magenta')

        resultado = np.log(matriz) / np.log(base)

        max_len = max(len(str(x)) for x in np.concatenate([matriz.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                print_colored(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animate_text(f"log({matriz[i,j]:>{max_len}}, {base}) = {resultado[i,j]:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        return resultado

class Division(MatrizOperaciones):
    """Class for matrix division operations."""

    @staticmethod
    def validar_matrices(matriz1, matriz2):
        """Validate that the second matrix is square and invertible."""
        if matriz2.shape[0] != matriz2.shape[1]:
            raise ValueError("La segunda matriz debe ser cuadrada")
        if abs(np.linalg.det(matriz2)) < 1e-10:
            raise ValueError("La segunda matriz no es invertible")
        return True

    @staticmethod
    def calcular_operacion(matriz1, matriz2):
        """Perform the division and show the step-by-step procedure."""
        Division.validar_matrices(matriz1, matriz2)

        print_colored(f"\n{'='*50}\n{'DIVISION DE MATRICES':^50}\n{'='*50}", 'magenta')

        det = np.linalg.det(matriz2)
        print_colored(f"\nDeterminante de la matriz: {det}", 'cyan')

        cofactores = np.zeros_like(matriz2, dtype=float)
        for i in range(matriz2.shape[0]):
            for j in range(matriz2.shape[1]):
                submatriz = np.delete(np.delete(matriz2, i, axis=0), j, axis=1)
                cofactores[i, j] = ((-1) ** (i + j)) * np.linalg.det(submatriz)

                print_colored(f"\nCofactor ({i+1},{j+1}): {cofactores[i,j]:^10}", 'cyan')
                time.sleep(0.2)  # Pause for movement effect

        adjunta = cofactores.T
        print_colored("\nMatriz de Cofactores:", 'cyan')
        MatrizUtil.imprimir_matriz(cofactores, "Matriz de Cofactores")
        print_colored("\nMatriz Adjunta (Transpuesta de la Matriz de Cofactores):", 'cyan')
        MatrizUtil.imprimir_matriz(adjunta, "Matriz Adjunta")

        inversa = adjunta / det
        print_colored("\nMatriz Inversa:", 'cyan')
        MatrizUtil.imprimir_matriz(inversa, "Matriz Inversa")

        resultado = np.dot(matriz1, inversa)
        print_colored("\nProcedimiento de multiplicación:", 'cyan')
        filas1, columnas1 = matriz1.shape
        filas2 = inversa.shape[1]

        max_len = max(len(str(x)) for x in np.concatenate([matriz1.flatten(), inversa.flatten(), resultado.flatten()]))
        formato = f"%{max_len}s"

        for i in range(filas1):
            for j in range(filas2):
                elementos = [f"({matriz1[i,k]} * {inversa[k,j]})" for k in range(columnas1)]
                expresion = " + ".join(elementos)
                resultado_elem = resultado[i,j]

                print_colored(f"\nElemento ({i+1},{j+1}):", 'cyan')
                animate_text(f"{expresion} = {resultado_elem:>{max_len}}", delay=0.01)
                time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        # Plot the result with separate visualizations
        MatrizUtil.plot_matriz_separate(matriz1, matriz2, resultado, "division")

        return resultado

class Sarrus(MatrizOperaciones):
    """Class to calculate the determinant of a 3x3 matrix using Sarrus' rule."""

    @staticmethod
    def validar_matriz(matriz):
        """Validate that the matrix is 3x3."""
        if matriz.shape != (3, 3):
            raise ValueError("La matriz debe ser de 3x3 para usar la regla de Sarrus")
        return True

    @staticmethod
    def calcular_determinante(matriz):
        """Calculate the determinant of a 3x3 matrix using Sarrus' rule."""
        Sarrus.validar_matriz(matriz)

        print_colored(f"\n{'='*50}\n{'DETERMINANTE POR SARRUS':^50}\n{'='*50}", 'magenta')

        a, b, c = matriz[0]
        d, e, f = matriz[1]
        g, h, i = matriz[2]

        # Calculate the determinant using Sarrus' rule
        determinante = (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (a * f * h) - (b * d * i)

        # Print the Sarrus operations
        animate_text(f"Determinante = ({a} * {e} * {i}) + ({b} * {f} * {g}) + ({c} * {d} * {h}) - ({c} * {e} * {g}) - ({a} * {f} * {h}) - ({b} * {d} * {i})", delay=0.01)
        print_colored(f"Determinante = {determinante}", 'cyan')
        time.sleep(0.5)  # Pause for movement effect

        # Calculate the cofactor matrix
        cofactores = np.zeros_like(matriz, dtype=float)
        for i in range(3):
            for j in range(3):
                submatriz = np.delete(np.delete(matriz, i, axis=0), j, axis=1)
                cofactores[i, j] = ((-1) ** (i + j)) * np.linalg.det(submatriz)
                print_colored(f"Cofactor ({i+1},{j+1}): (-1)^{i+j} * det(submatriz) = {cofactores[i,j]}", 'cyan')
                time.sleep(0.2)  # Pause for movement effect

        # Calculate the adjugate matrix (transpose of the cofactor matrix)
        adjunta = cofactores.T

        # Calculate the inverse matrix
        inversa = adjunta / determinante

        # Print the matrices
        print_colored("\nMatriz de Cofactores:", 'cyan')
        MatrizUtil.imprimir_matriz(cofactores, "Matriz de Cofactores")
        print_colored("\nMatriz Adjunta (Transpuesta de la Matriz de Cofactores):", 'cyan')
        MatrizUtil.imprimir_matriz(adjunta, "Matriz Adjunta")
        print_colored("\nMatriz Inversa:", 'cyan')
        MatrizUtil.imprimir_matriz(inversa, "Matriz Inversa")

        # Plot the determinant with separate visualizations
        MatrizUtil.plot_determinante_separate(matriz, determinante)

        return determinante

class GaussJordan(MatrizOperaciones):
    """Class to solve systems of equations using the Gauss-Jordan method."""

    @staticmethod
    def validar_matriz(matriz):
        """Validate that the matrix is square."""
        if matriz.shape[0] != matriz.shape[1]:
            raise ValueError("La matriz debe ser cuadrada para usar Gauss-Jordan")
        return True

    @staticmethod
    def calcular_gauss_jordan(matriz):
        """Perform Gauss-Jordan elimination and show the step-by-step procedure."""
        GaussJordan.validar_matriz(matriz)

        print_colored(f"\n{'='*50}\n{'GAUSS-JORDAN':^50}\n{'='*50}", 'magenta')

        matriz_aumentada = np.hstack([matriz, np.identity(matriz.shape[0], dtype=np.float64)])
        n = matriz.shape[0]

        ecuaciones = []
        pasos_matrices = [matriz_aumentada.copy()]  # Store each step of the matrix

        for i in range(n):
            # Make the pivot 1
            if matriz_aumentada[i, i] != 1:
                factor = 1 / matriz_aumentada[i, i]
                matriz_aumentada[i] *= factor
                ecuaciones.append(f"Fila {i+1}: Dividir cada elemento por {factor:.2f} para hacer el pivote 1")
                animate_text(ecuaciones[-1], 'cyan')
                pasos_matrices.append(matriz_aumentada.copy())
                time.sleep(0.2)  # Pause for movement effect

            # Make zeros below and above the pivot
            for j in range(n):
                if i != j:
                    factor = matriz_aumentada[j, i]
                    matriz_aumentada[j] -= factor * matriz_aumentada[i]
                    ecuaciones.append(f"Fila {j+1}: Restar {factor:.2f} veces la fila {i+1} de la fila {j+1}")
                    animate_text(ecuaciones[-1], 'cyan')
                    pasos_matrices.append(matriz_aumentada.copy())
                    time.sleep(0.2)  # Pause for movement effect

        inversa = matriz_aumentada[:, n:]
        print_colored("\nDesarrollo de las ecuaciones:", 'cyan')
        for eq in ecuaciones:
            animate_text(eq, 'cyan')
            time.sleep(0.2)  # Pause for movement effect

        print_colored("\nMatriz Inversa:", 'cyan')
        MatrizUtil.imprimir_matriz(inversa, "Matriz Inversa")

        # Plot the Gauss-Jordan process with separate visualizations
        MatrizUtil.plot_gauss_jordan_separate(matriz, inversa, pasos_matrices)

        return inversa

class VectorOperaciones:
    """Class for vector operations."""

    @staticmethod
    def validar_vectores(vector1, vector2=None):
        """Validate the dimensions of the vectors according to the operation."""
        if vector2 is None:
            return vector1.size > 0
        return vector1.size > 0 and vector2.size > 0 and vector1.shape == vector2.shape

    @staticmethod
    def suma_vectores(vector1, vector2):
        """Perform vector addition and show the step-by-step procedure."""
        VectorOperaciones.validar_vectores(vector1, vector2)

        print_colored(f"\n{'='*50}\n{'SUMA DE VECTORES':^50}\n{'='*50}", 'magenta')

        resultado = vector1 + vector2

        max_len = max(len(str(x)) for x in np.concatenate([vector1.flatten(), vector2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector1.shape[0]):
            print_colored(f"\nElemento ({i+1}):", 'cyan')
            animate_text(f"{vector1[i]:>{max_len}} + {vector2[i]:>{max_len}} = {resultado[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        return resultado

    @staticmethod
    def resta_vectores(vector1, vector2):
        """Perform vector subtraction and show the step-by-step procedure."""
        VectorOperaciones.validar_vectores(vector1, vector2)

        print_colored(f"\n{'='*50}\n{'RESTA DE VECTORES':^50}\n{'='*50}", 'magenta')

        resultado = vector1 - vector2

        max_len = max(len(str(x)) for x in np.concatenate([vector1.flatten(), vector2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector1.shape[0]):
            print_colored(f"\nElemento ({i+1}):", 'cyan')
            animate_text(f"{vector1[i]:>{max_len}} - {vector2[i]:>{max_len}} = {resultado[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        return resultado

    @staticmethod
    def producto_escalar(vector1, vector2):
        """Perform scalar product (dot product) and show the step-by-step procedure."""
        VectorOperaciones.validar_vectores(vector1, vector2)

        print_colored(f"\n{'='*50}\n{'PRODUCTO ESCALAR':^50}\n{'='*50}", 'magenta')

        resultado = np.dot(vector1, vector2)

        max_len = max(len(str(x)) for x in np.concatenate([vector1.flatten(), vector2.flatten()]))
        formato = f"%{max_len}s"

        for i in range(vector1.shape[0]):
            print_colored(f"\nElemento ({i+1}):", 'cyan')
            animate_text(f"{vector1[i]:>{max_len}} * {vector2[i]:>{max_len}} = {vector1[i] * vector2[i]:>{max_len}}", delay=0.01)
            time.sleep(0.2)  # Pause for movement effect

        print_colored("\nResultado:", 'cyan')
        print_colored(f"Producto Escalar: {resultado}", 'cyan')

        return resultado

    @staticmethod
    def producto_vectorial(vector1, vector2):
        """Perform vector cross product and show the step-by-step procedure."""
        if vector1.shape[0] != 3 or vector2.shape[0] != 3:
            raise ValueError("Los vectores deben tener 3 componentes para el producto vectorial")

        print_colored(f"\n{'='*50}\n{'PRODUCTO VECTORIAL':^50}\n{'='*50}", 'magenta')

        resultado = np.cross(vector1, vector2)

        print_colored(f"Vector 1: {vector1}", 'cyan')
        print_colored(f"Vector 2: {vector2}", 'cyan')
        print_colored("\nResultado:", 'cyan')
        MatrizUtil.imprimir_matriz(resultado, "Resultado")

        return resultado

    @staticmethod
    def magnitud_vector(vector):
        """Calculate the magnitude of a vector and show the step-by-step procedure."""
        VectorOperaciones.validar_vectores(vector)

        print_colored(f"\n{'='*50}\n{'MAGNITUD DEL VECTOR':^50}\n{'='*50}", 'magenta')

        resultado = np.linalg.norm(vector)

        print_colored(f"Vector: {vector}", 'cyan')
        print_colored("\nResultado:", 'cyan')
        print_colored(f"Magnitud: {resultado}", 'cyan')

        return resultado

    @staticmethod
    def angulo_entre_vectores(vector1, vector2):
        """Calculate the angle between two vectors and show the step-by-step procedure."""
        VectorOperaciones.validar_vectores(vector1, vector2)

        print_colored(f"\n{'='*50}\n{'ANGULO ENTRE VECTORES':^50}\n{'='*50}", 'magenta')

        dot_product = np.dot(vector1, vector2)
        magnitud1 = np.linalg.norm(vector1)
        magnitud2 = np.linalg.norm(vector2)

        resultado = np.arccos(dot_product / (magnitud1 * magnitud2))
        resultado_deg = np.degrees(resultado)

        print_colored(f"Vector 1: {vector1}", 'cyan')
        print_colored(f"Vector 2: {vector2}", 'cyan')
        print_colored("\nResultado:", 'cyan')
        print_colored(f"Ángulo: {resultado_deg:.2f} grados", 'cyan')

        return resultado_deg

class ComprehensiveOperations(MatrizOperaciones):
    """Class to perform all matrix operations on the same matrices."""

    @staticmethod
    def realizar_todas_operaciones(matriz1, matriz2):
        """Perform all valid operations on the given matrices."""
        print_colored(f"\n{'='*50}\n{'ANÁLISIS COMPLETO DE MATRICES':^50}\n{'='*50}", 'magenta')

        operaciones_realizadas = []
        resultados = {}

        # Try to perform all operations
        try:
            # Suma
            print_colored("\n1. SUMA DE MATRICES", 'yellow')
            resultados['suma'] = SumaResta.calcular_operacion(matriz1, matriz2, "suma")
            operaciones_realizadas.append("suma")
        except ValueError as e:
            print_colored(f"No se pudo realizar la suma: {e}", 'red')

        try:
            # Resta
            print_colored("\n2. RESTA DE MATRICES", 'yellow')
            resultados['resta'] = SumaResta.calcular_operacion(matriz1, matriz2, "resta")
            operaciones_realizadas.append("resta")
        except ValueError as e:
            print_colored(f"No se pudo realizar la resta: {e}", 'red')

        try:
            # Multiplicación
            print_colored("\n3. MULTIPLICACIÓN DE MATRICES", 'yellow')
            resultados['multiplicacion'] = Multiplicacion.calcular_operacion(matriz1, matriz2)
            operaciones_realizadas.append("multiplicacion")
        except ValueError as e:
            print_colored(f"No se pudo realizar la multiplicación: {e}", 'red')

        try:
            # División
            print_colored("\n4. DIVISIÓN DE MATRICES", 'yellow')
            resultados['division'] = Division.calcular_operacion(matriz1, matriz2)
            operaciones_realizadas.append("division")
        except ValueError as e:
            print_colored(f"No se pudo realizar la división: {e}", 'red')

        # Determinantes (si son matrices cuadradas)
        if matriz1.shape[0] == matriz1.shape[1]:
            try:
                print_colored(f"\n5. DETERMINANTE DE MATRIZ 1", 'yellow')
                determinante1 = np.linalg.det(matriz1)
                print_colored(f"Determinante: {determinante1}", 'cyan')
                resultados['determinante1'] = determinante1

                # Si es 3x3, usar Sarrus
                if matriz1.shape == (3, 3):
                    print_colored("\nUsando Regla de Sarrus:", 'yellow')
                    Sarrus.calcular_determinante(matriz1)

                operaciones_realizadas.append("determinante1")
            except Exception as e:
                print_colored(f"No se pudo calcular el determinante de matriz 1: {e}", 'red')

            try:
                # Gauss-Jordan para matriz 1
                print_colored(f"\n6. GAUSS-JORDAN PARA MATRIZ 1", 'yellow')
                resultados['inversa1'] = GaussJordan.calcular_gauss_jordan(matriz1)
                operaciones_realizadas.append("gauss_jordan1")
            except Exception as e:
                print_colored(f"No se pudo aplicar Gauss-Jordan a matriz 1: {e}", 'red')

        if matriz2.shape[0] == matriz2.shape[1]:
            try:
                print_colored(f"\n7. DETERMINANTE DE MATRIZ 2", 'yellow')
                determinante2 = np.linalg.det(matriz2)
                print_colored(f"Determinante: {determinante2}", 'cyan')
                resultados['determinante2'] = determinante2

                # Si es 3x3, usar Sarrus
                if matriz2.shape == (3, 3):
                    print_colored("\nUsando Regla de Sarrus:", 'yellow')
                    Sarrus.calcular_determinante(matriz2)

                operaciones_realizadas.append("determinante2")
            except Exception as e:
                print_colored(f"No se pudo calcular el determinante de matriz 2: {e}", 'red')

            try:
                # Gauss-Jordan para matriz 2
                print_colored(f"\n8. GAUSS-JORDAN PARA MATRIZ 2", 'yellow')
                resultados['inversa2'] = GaussJordan.calcular_gauss_jordan(matriz2)
                operaciones_realizadas.append("gauss_jordan2")
            except Exception as e:
                print_colored(f"No se pudo aplicar Gauss-Jordan a matriz 2: {e}", 'red')

        # Resumen de todas las operaciones realizadas
        print_colored(f"\n{'='*50}\n{'RESUMEN DE OPERACIONES':^50}\n{'='*50}", 'magenta')
        for op in operaciones_realizadas:
            if op == 'determinante1':
                print_colored(f"- Determinante de Matriz 1: {resultados[op]}", 'green')
            elif op == 'determinante2':
                print_colored(f"- Determinante de Matriz 2: {resultados[op]}", 'green')
            else:
                print_colored(f"- {op.capitalize()} realizada correctamente", 'green')

        return resultados

class MatrizUtil:
    """Utility class for common matrix operations."""

    @staticmethod
    def ingresar_matriz():
        """Allow the user to enter the elements of a matrix interactively."""
        try:
            filas = int(input("Ingrese el número de filas: "))
            columnas = int(input("Ingrese el número de columnas: "))

            print_colored(f"Ingrese los {filas * columnas} elementos separados por espacios (fila por fila):", 'cyan')
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
                        print_colored(f"Error: {e}", 'red')

            matriz = np.array(elementos)
            print_colored("Matriz ingresada:", 'green')
            MatrizUtil.imprimir_matriz(matriz)
            confirmacion = input("¿Es correcta la matriz ingresada? (s/n): ").strip().lower()
            if confirmacion != 's':
                return MatrizUtil.ingresar_matriz()

            return matriz

        except ValueError as e:
            print_colored(f"Error al ingresar la matriz: {str(e)}", 'red')
            return np.array([])

    @staticmethod
    def generar_matriz_aleatoria():
        """Generate a random matrix with user-defined dimensions."""
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

            print_colored("Matriz generada:", 'green')
            MatrizUtil.imprimir_matriz(matriz)

            return matriz

        except ValueError as e:
            print_colored(f"Error al generar la matriz: {str(e)}", 'red')
            return np.array([])

    @staticmethod
    def ingresar_vector():
        """Allow the user to enter the elements of a vector interactively."""
        try:
            dimension = int(input("Ingrese la dimensión del vector: "))

            print_colored(f"Ingrese los {dimension} elementos separados por espacios:", 'cyan')
            elementos = list(map(float, input().split()))

            if len(elementos) != dimension:
                raise ValueError(f"Número incorrecto de elementos. Se esperaban {dimension} elementos.")

            vector = np.array(elementos)
            print_colored("Vector ingresado:", 'green')
            MatrizUtil.imprimir_matriz(vector)
            confirmacion = input("¿Es correcto el vector ingresado? (s/n): ").strip().lower()
            if confirmacion != 's':
                return MatrizUtil.ingresar_vector()

            return vector

        except ValueError as e:
            print_colored(f"Error al ingresar el vector: {str(e)}", 'red')
            return np.array([])

    @staticmethod
    def generar_vector_aleatorio():
        """Generate a random vector with user-defined dimension."""
        try:
            dimension = int(input("Ingrese la dimensión del vector: "))

            min_val = float(input("Ingrese el valor mínimo para los elementos: "))
            max_val = float(input("Ingrese el valor máximo para los elementos: "))

            vector = np.random.uniform(low=min_val, high=max_val, size=dimension)

            # Opción para redondear a enteros
            redondear = input("¿Desea redondear a números enteros? (s/n): ").strip().lower()
            if redondear == 's':
                vector = np.round(vector).astype(int)

            print_colored("Vector generado:", 'green')
            MatrizUtil.imprimir_matriz(vector)

            return vector

        except ValueError as e:
            print_colored(f"Error al generar el vector: {str(e)}", 'red')
            return np.array([])

    @staticmethod
    def imprimir_matriz(matriz, titulo=""):
        """Print a matrix with structured formatting."""
        if matriz.size == 0:
            print("(Matriz vacía)")
            return

        max_len = max(len(str(x)) for x in matriz.flatten())
        formato = f"%{max_len}s"

        if titulo:
            print_colored(f"\n{titulo:^50}", 'yellow')

        print("-" * (matriz.shape[1] * (max_len + 2) + 2))
        for fila in matriz:
            print("| " + " ".join(formato % (str(int(x)) if x.is_integer() else str(x)) for x in fila) + " |")
        print("-" * (matriz.shape[1] * (max_len + 2) + 2))

    @staticmethod
    def imprimir_matrices_lado_a_lado(matriz1, matriz2, titulo=""):
        """Print two matrices side by side with structured formatting."""
        if matriz1.size == 0 or matriz2.size == 0:
            print("(Matriz vacía)")
            return

        max_len1 = max(len(str(x)) for x in matriz1.flatten())
        max_len2 = max(len(str(x)) for x in matriz2.flatten())
        formato1 = f"%{max_len1}s"
        formato2 = f"%{max_len2}s"

        max_filas = max(matriz1.shape[0], matriz2.shape[0])

        print_colored(f"\n{'='*50}\n{titulo:^50}\n{'='*50}", 'yellow')
        print_colored(f"{'Matriz 1':^{max_len1 * matriz1.shape[1] + 2}} | {'Matriz 2':^{max_len2 * matriz2.shape[1] + 2}}", 'cyan')
        print("-" * ((max_len1 + 2) * matriz1.shape[1] + 3 + (max_len2 + 2) * matriz2.shape[1]))

        for i in range(max_filas):
            fila1 = " ".join(formato1 % (str(int(x)) if x.is_integer() else str(x)) for x in matriz1[i]) if i < matriz1.shape[0] else " " * ((max_len1 + 2) * matriz1.shape[1])
            fila2 = " ".join(formato2 % (str(int(x)) if x.is_integer() else str(x)) for x in matriz2[i]) if i < matriz2.shape[0] else " " * ((max_len2 + 2) * matriz2.shape[1])
            print_colored(f"| {fila1} | | {fila2} |", 'cyan')
            time.sleep(0.2)  # Pause for movement effect

    @staticmethod
    def normalize_data_for_pie(data):
        """Normalize data for pie chart (make all values positive and ensure sum > 0)."""
        abs_data = np.abs(data)
        if np.sum(abs_data) == 0:
            # If all values are zero, create dummy data
            return np.ones_like(abs_data)
        return abs_data

    @staticmethod
    def create_labels(matriz, prefix=''):
        """Create labels for the data elements."""
        rows, cols = matriz.shape
        return [f"{prefix}({i+1},{j+1}): {matriz[i,j]}" for i in range(rows) for j in range(cols)]

    @staticmethod
    def plot_matriz_separate(matriz1, matriz2, resultado, operacion):
        """Plot the matrices and the result as separate charts."""
        # Get flattened data
        matriz1_flat = matriz1.flatten()
        matriz2_flat = matriz2.flatten()
        resultado_flat = resultado.flatten()

        titles = ['Matriz 1', 'Matriz 2', 'Resultado']
        data_sets = [matriz1_flat, matriz2_flat, resultado_flat]
        colors = ['skyblue', 'lightgreen', 'salmon']

        # Create bar charts (separate figures)
        for i, (title, data, color) in enumerate(zip(titles, data_sets, colors)):
            plt.figure(figsize=(10, 6))
            indices = range(len(data))
            plt.bar(indices, data, color=color, alpha=0.7)
            plt.title(f'{title}', fontsize=14)
            plt.xticks(indices, [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in indices] if len(data) <= 20 else [])
            if len(data) > 20:
                plt.xlabel(f"{len(data)} elementos", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Create line charts (separate figures)
        for i, (title, data, color) in enumerate(zip(titles, data_sets, colors)):
            plt.figure(figsize=(10, 6))
            indices = range(len(data))
            plt.plot(indices, data, marker='o', linestyle='-', color=color)
            plt.title(f'{title}', fontsize=14)
            plt.xticks(indices, [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in indices] if len(data) <= 20 else [])
            if len(data) > 20:
                plt.xlabel(f"{len(data)} elementos", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Create histograms (separate figures)
        for i, (title, data, color) in enumerate(zip(titles, data_sets, colors)):
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=min(10, len(data)), color=color, alpha=0.7, edgecolor='black')
            plt.title(f'{title}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Create scatter plots (separate figures)
        for i, (title, data, color) in enumerate(zip(titles, data_sets, colors)):
            plt.figure(figsize=(10, 6))
            indices = range(len(data))
            plt.scatter(indices, data, color=color, alpha=0.7)
            plt.title(f'{title}', fontsize=14)
            plt.xticks(indices, [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in indices] if len(data) <= 20 else [])
            if len(data) > 20:
                plt.xlabel(f"{len(data)} elementos", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Create pie charts (separate figures)
        for i, (title, data, color) in enumerate(zip(titles, data_sets, colors)):
            plt.figure(figsize=(10, 6))
            pie_data = MatrizUtil.normalize_data_for_pie(data)
            if len(data) <= 10:  # Only show detailed labels for smaller matrices
                labels = [f'({i//matriz1.shape[1]+1},{i%matriz1.shape[1]+1})' for i in range(len(data))]
                plt.pie(pie_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(data))))
            else:
                plt.pie(pie_data, startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(data))))
                plt.text(0, 0, f"{len(data)} elementos", ha='center', va='center', fontsize=12)

            plt.title(f'{title}', fontsize=14)
            plt.suptitle(f'Operación: {operacion.capitalize()}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    @staticmethod
    def plot_determinante_separate(matriz, determinante):
        """Plot the matrix and its determinant as separate charts."""
        plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=plt.gcf())

        # Get flattened matrix data
        matriz_flat = matriz.flatten()

        # Bar chart
        ax1 = plt.subplot(gs[0, 0])
        indices = range(len(matriz_flat))
        ax1.bar(indices, matriz_flat, color='lightcoral', alpha=0.7)
        ax1.set_title('Matriz', fontsize=12)
        ax1.set_xticks(indices)
        ax1.set_xticklabels([f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in indices], fontsize=8, rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Line chart
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(matriz_flat, marker='o', linestyle='-', color='lightcoral')
        ax2.set_title('Matriz', fontsize=12)
        ax2.set_xticks(indices)
        ax2.set_xticklabels([f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in indices], fontsize=8, rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Scatter plot
        ax3 = plt.subplot(gs[0, 2])
        ax3.scatter(indices, matriz_flat, color='lightcoral', alpha=0.7)
        ax3.set_title('Matriz', fontsize=12)
        ax3.set_xticks(indices)
        ax3.set_xticklabels([f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in indices], fontsize=8, rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.6)

        # Histogram
        ax4 = plt.subplot(gs[1, 0])
        ax4.hist(matriz_flat, bins=min(10, len(matriz_flat)), color='lightcoral', alpha=0.7, edgecolor='black')
        ax4.set_title('Matriz', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.6)

        # Pie chart for matrix elements
        ax5 = plt.subplot(gs[1, 1])
        pie_data = MatrizUtil.normalize_data_for_pie(matriz_flat)
        labels = [f'({i//matriz.shape[1]+1},{i%matriz.shape[1]+1})' for i in range(len(matriz_flat))]
        ax5.pie(pie_data, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(matriz_flat))))
        ax5.set_title('Matriz', fontsize=12)

        # Visualization of the determinant
        ax6 = plt.subplot(gs[1, 2])
        # Create a visual representation of the determinant
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.bar(['Determinante'], [determinante], color='purple' if determinante >= 0 else 'red', alpha=0.7)
        ax6.set_title(f'Determinante: {determinante}', fontsize=14)
        ax6.grid(True, linestyle='--', alpha=0.6)
        # Add text annotation showing the value
        ax6.text(0, determinante/2, f"{determinante:.2f}", ha='center', fontsize=14,
                 color='white' if abs(determinante) > 3 else 'black')

        plt.suptitle(f'Análisis del Determinante de la Matriz', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_gauss_jordan_separate(matriz_original, inversa, pasos_matrices):
        """Plot the Gauss-Jordan elimination process as separate charts."""
        # Determine how many visualizations to show
        num_pasos = min(4, len(pasos_matrices))  # Show at most 4 steps
        indices = np.linspace(0, len(pasos_matrices)-1, num_pasos, dtype=int)

        # Plot original matrix
        matriz_flat = matriz_original.flatten()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(matriz_flat)), matriz_flat, color='royalblue', alpha=0.7)
        plt.title('Matriz Original', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Plot original matrix as scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(matriz_flat)), matriz_flat, color='royalblue', alpha=0.7)
        plt.title('Matriz Original', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Plot original matrix as histogram
        plt.figure(figsize=(10, 6))
        plt.hist(matriz_flat, bins=min(10, len(matriz_flat)), color='royalblue',
                 alpha=0.7, edgecolor='black')
        plt.title('Matriz Original', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Plot selected steps of the elimination process
        for i, idx in enumerate(indices):
            if i == len(indices) - 1:  # Skip the last one, we'll show the final result separately
                continue

            paso_matriz = pasos_matrices[idx]
            paso_matriz_flat = paso_matriz.flatten()

            # Bar chart for this step
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(paso_matriz_flat)), paso_matriz_flat,
                    color='lightgreen', alpha=0.7)
            plt.title(f'Paso {idx+1}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            # Scatter plot for this step
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(paso_matriz_flat)), paso_matriz_flat,
                    color='lightgreen', alpha=0.7)
            plt.title(f'Paso {idx+1}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            # Histogram for this step
            plt.figure(figsize=(10, 6))
            plt.hist(paso_matriz_flat, bins=min(10, len(paso_matriz_flat)),
                     color='lightgreen', alpha=0.7, edgecolor='black')
            plt.title(f'Paso {idx+1}', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Plot the final inverse matrix
        inversa_flat = inversa.flatten()
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(inversa_flat)), inversa_flat, color='gold', alpha=0.7)
        plt.title('Matriz Inversa', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Scatter plot for the final inverse matrix
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(inversa_flat)), inversa_flat, color='gold', alpha=0.7)
        plt.title('Matriz Inversa', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Histogram for the final inverse matrix
        plt.figure(figsize=(10, 6))
        plt.hist(inversa_flat, bins=min(10, len(inversa_flat)),
                 color='gold', alpha=0.7, edgecolor='black')
        plt.title('Matriz Inversa', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def main():
    clear_console()
    animate_title("Calculadora de Matrices y Vectores")

    while True:
        tipo_operacion = input("¿Desea trabajar con matrices o vectores? (m/v): ").strip().lower()

        if tipo_operacion == 'm':
            print_colored("\nIngrese el número de filas y columnas de la primera matriz:", 'cyan')
            filas1 = int(input("Ingrese el número de filas: "))
            columnas1 = int(input("Ingrese el número de columnas: "))

            print_colored(f"Ingrese los {filas1 * columnas1} elementos separados por espacios (fila por fila):", 'cyan')
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
                        print_colored(f"Error: {e}", 'red')

            matriz1 = np.array(elementos1)
            print_colored("Primera matriz ingresada:", 'green')
            MatrizUtil.imprimir_matriz(matriz1)

            print_colored("\nIngrese el número de filas y columnas de la segunda matriz:", 'cyan')
            filas2 = int(input("Ingrese el número de filas: "))
            columnas2 = int(input("Ingrese el número de columnas: "))

            print_colored(f"Ingrese los {filas2 * columnas2} elementos separados por espacios (fila por fila):", 'cyan')
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
                        print_colored(f"Error: {e}", 'red')

            matriz2 = np.array(elementos2)
            print_colored("Segunda matriz ingresada:", 'green')
            MatrizUtil.imprimir_matriz(matriz2)

            print_colored("\nSeleccione una operación:", 'cyan')
            print_colored("1. Suma de matrices", 'yellow')
            print_colored("2. Resta de matrices", 'yellow')
            print_colored("3. Multiplicación de matrices", 'yellow')
            print_colored("4. División de matrices", 'yellow')
            print_colored("5. Determinante por Sarrus (3x3)", 'yellow')
            print_colored("6. Gauss-Jordan", 'yellow')
            print_colored("7. Multiplicación escalar-vector", 'yellow')
            print_colored("8. Potencia de matriz", 'yellow')
            print_colored("9. Raíz de matriz", 'yellow')
            print_colored("10. Logaritmo de matriz", 'yellow')
            print_colored("11. Multiplicación matriz A1X1", 'yellow')
            print_colored("12. Análisis completo de matrices", 'yellow')
            print_colored("13. Salir", 'yellow')

            opcion = input("Ingrese el número de la operación deseada: ").strip()

            if opcion == '1':
                resultado = SumaResta.calcular_operacion(matriz1, matriz2, "suma")
                MatrizUtil.imprimir_matriz(resultado, "Resultado de la Suma")

            elif opcion == '2':
                resultado = SumaResta.calcular_operacion(matriz1, matriz2, "resta")
                MatrizUtil.imprimir_matriz(resultado, "Resultado de la Resta")

            elif opcion == '3':
                resultado = Multiplicacion.calcular_operacion(matriz1, matriz2)
                MatrizUtil.imprimir_matriz(resultado, "Resultado de la Multiplicación")

            elif opcion == '4':
                resultado = Division.calcular_operacion(matriz1, matriz2)
                MatrizUtil.imprimir_matriz(resultado, "Resultado de la División")

            elif opcion == '5':
                if matriz1.shape == (3, 3):
                    resultado = Sarrus.calcular_determinante(matriz1)
                    print_colored(f"Determinante de la matriz 1: {resultado}", 'cyan')
                else:
                    print_colored("La matriz 1 no es 3x3, no se puede aplicar la regla de Sarrus.", 'red')

                if matriz2.shape == (3, 3):
                    resultado = Sarrus.calcular_determinante(matriz2)
                    print_colored(f"Determinante de la matriz 2: {resultado}", 'cyan')
                else:
                    print_colored("La matriz 2 no es 3x3, no se puede aplicar la regla de Sarrus.", 'red')

            elif opcion == '6':
                if matriz1.shape[0] == matriz1.shape[1]:
                    resultado = GaussJordan.calcular_gauss_jordan(matriz1)
                    MatrizUtil.imprimir_matriz(resultado, "Matriz Inversa de la Matriz 1")
                else:
                    print_colored("La matriz 1 no es cuadrada, no se puede aplicar Gauss-Jordan.", 'red')

                if matriz2.shape[0] == matriz2.shape[1]:
                    resultado = GaussJordan.calcular_gauss_jordan(matriz2)
                    MatrizUtil.imprimir_matriz(resultado, "Matriz Inversa de la Matriz 2")
                else:
                    print_colored("La matriz 2 no es cuadrada, no se puede aplicar Gauss-Jordan.", 'red')

            elif opcion == '7':
                escalar = float(input("Ingrese el valor del escalar: "))
                vector = np.array(list(map(float, input("Ingrese los elementos del vector separados por espacios: ").split())))
                resultado = EscalarVector.calcular_operacion(escalar, vector)
                MatrizUtil.imprimir_matriz(resultado, "Resultado de la Multiplicación Escalar-Vector")

            elif opcion == '8':
                exponente = float(input("Ingrese el valor del exponente: "))
                resultado1 = PotenciaRaizLogaritmo.calcular_potencia(matriz1, exponente)
                MatrizUtil.imprimir_matriz(resultado1, "Resultado de la Potencia de la Matriz 1")
                resultado2 = PotenciaRaizLogaritmo.calcular_potencia(matriz2, exponente)
                MatrizUtil.imprimir_matriz(resultado2, "Resultado de la Potencia de la Matriz 2")

            elif opcion == '9':
                raiz = float(input("Ingrese el valor de la raíz: "))
                resultado1 = PotenciaRaizLogaritmo.calcular_raiz(matriz1, raiz)
                MatrizUtil.imprimir_matriz(resultado1, "Resultado de la Raíz de la Matriz 1")
                resultado2 = PotenciaRaizLogaritmo.calcular_raiz(matriz2, raiz)
                MatrizUtil.imprimir_matriz(resultado2, "Resultado de la Raíz de la Matriz 2")

            elif opcion == '10':
                base = float(input("Ingrese el valor de la base del logaritmo: "))
                resultado1 = PotenciaRaizLogaritmo.calcular_logaritmo(matriz1, base)
                MatrizUtil.imprimir_matriz(resultado1, "Resultado del Logaritmo de la Matriz 1")
                resultado2 = PotenciaRaizLogaritmo.calcular_logaritmo(matriz2, base)
                MatrizUtil.imprimir_matriz(resultado2, "Resultado del Logaritmo de la Matriz 2")

            elif opcion == '11':
                if matriz1.shape[1] == 1:
                    vector = np.array(list(map(float, input("Ingrese los elementos del vector separados por espacios: ").split())))
                    if vector.shape[0] == matriz1.shape[0]:
                        resultado = np.dot(matriz1, vector)
                        MatrizUtil.imprimir_matriz(resultado, "Resultado de la Multiplicación A1X1")
                    else:
                        print_colored("El vector no tiene la misma cantidad de filas que la matriz.", 'red')
                else:
                    print_colored("La matriz 1 no tiene una sola columna.", 'red')

            elif opcion == '12':
                resultados = ComprehensiveOperations.realizar_todas_operaciones(matriz1, matriz2)

            elif opcion == '13':
                print_colored("Saliendo del programa...", 'red')
                break

            else:
                print_colored("Opción no válida. Por favor, intente de nuevo.", 'red')

        elif tipo_operacion == 'v':
            print_colored("\nIngrese la dimensión del primer vector:", 'cyan')
            dimension1 = int(input("Ingrese la dimensión: "))

            print_colored(f"Ingrese los {dimension1} elementos separados por espacios:", 'cyan')
            vector1 = MatrizUtil.ingresar_vector()

            print_colored("\nIngrese la dimensión del segundo vector:", 'cyan')
            dimension2 = int(input("Ingrese la dimensión: "))

            print_colored(f"Ingrese los {dimension2} elementos separados por espacios:", 'cyan')
            vector2 = MatrizUtil.ingresar_vector()

            print_colored("\nSeleccione una operación:", 'cyan')
            print_colored("1. Suma de vectores", 'yellow')
            print_colored("2. Resta de vectores", 'yellow')
            print_colored("3. Producto escalar", 'yellow')
            print_colored("4. Producto vectorial", 'yellow')
            print_colored("5. Magnitud del vector", 'yellow')
            print_colored("6. Ángulo entre vectores", 'yellow')
            print_colored("7. Salir", 'yellow')

            opcion = input("Ingrese el número de la operación deseada: ").strip()

            if opcion == '1':
                resultado = VectorOperaciones.suma_vectores(vector1, vector2)
                MatrizUtil.imprimir_matriz(resultado, "Resultado de la Suma de Vectores")

            elif opcion == '2':
                resultado = VectorOperaciones.resta_vectores(vector1, vector2)
                MatrizUtil.imprimir_matriz(resultado, "Resultado de la Resta de Vectores")

            elif opcion == '3':
                resultado = VectorOperaciones.producto_escalar(vector1, vector2)
                print_colored(f"Producto Escalar: {resultado}", 'cyan')

            elif opcion == '4':
                resultado = VectorOperaciones.producto_vectorial(vector1, vector2)
                MatrizUtil.imprimir_matriz(resultado, "Resultado del Producto Vectorial")

            elif opcion == '5':
                resultado1 = VectorOperaciones.magnitud_vector(vector1)
                print_colored(f"Magnitud del Vector 1: {resultado1}", 'cyan')
                resultado2 = VectorOperaciones.magnitud_vector(vector2)
                print_colored(f"Magnitud del Vector 2: {resultado2}", 'cyan')

            elif opcion == '6':
                resultado = VectorOperaciones.angulo_entre_vectores(vector1, vector2)
                print_colored(f"Ángulo entre Vectores: {resultado} grados", 'cyan')

            elif opcion == '7':
                print_colored("Saliendo del programa...", 'red')
                break

            else:
                print_colored("Opción no válida. Por favor, intente de nuevo.", 'red')

        else:
            print_colored("Opción no válida. Por favor, intente de nuevo.", 'red')

if __name__ == "__main__":
    main()
