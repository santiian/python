import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from dataclasses import dataclass, field
import threading
from typing import Tuple, List, Optional, Dict, Any, Callable, TypeAlias, Final
from enum import Enum, auto
import sv_ttk
import datetime
import random
from typing import TypeAlias, Final, Dict, Any, Optional, Tuple, List, Callable
import typing_extensions

# Definiciones de clases faltantes
class Tema(Enum):
    OSCURO = auto()
    CLARO = auto()

class ModoVista(Enum):
    MODO_2D = auto()
    MODO_3D = auto()

class CondicionClima(Enum):
    DESPEJADO = auto()
    LLUVIOSO = auto()
    NUBLADO = auto()

class EstadoSimulacion(Enum):
    INICIAL = auto()
    EN_EJECUCION = auto()
    ANIMANDO = auto()
    COMPLETADO = auto()

@dataclass
class ParametrosSimulacion:
    altura_enemigo: float = 10000.0
    distancia_defensa: float = 30000.0
    velocidad_antimisil: float = 1000.0
    angulo_antimisil: float = 45.0
    tiempo_inicio_antimisil: float = 5.0
    velocidad_animacion: float = 1.0
    modo_vista: ModoVista = ModoVista.MODO_2D
    clima: CondicionClima = CondicionClima.DESPEJADO

    @dataclass
    class Constantes:
        GRAVEDAD: float = 9.81
        PASO_TIEMPO: float = 0.1
        MARGEN_ERROR: float = 50.0

    constantes: Constantes = field(default_factory=Constantes)

@dataclass
class ResultadosSimulacion:
    intercepcion: bool = False
    tiempo_intercepcion: float = 0.0
    distancia_minima: float = float('inf')
    interceptar_x: Optional[float] = None
    interceptar_y: Optional[float] = None
    interceptar_z: Optional[float] = None

class CalculadorTrayectorias:
    @staticmethod
    def calcular_trayectorias(params: ParametrosSimulacion) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Implementación simplificada
        t = np.arange(0, 20, params.constantes.PASO_TIEMPO)
        enemigo_x = np.linspace(params.distancia_defensa, 0, len(t))
        enemigo_y = np.linspace(params.altura_enemigo, 0, len(t))
        antimisil_x = np.linspace(0, params.distancia_defensa, len(t))
        antimisil_y = np.linspace(0, params.altura_enemigo, len(t))
        return t, enemigo_x, enemigo_y, antimisil_x, antimisil_y

    @staticmethod
    def calcular_trayectorias_3d(params: ParametrosSimulacion) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Implementación simplificada
        t = np.arange(0, 20, params.constantes.PASO_TIEMPO)
        enemigo_x = np.linspace(params.distancia_defensa, 0, len(t))
        enemigo_y = np.linspace(0, 0, len(t))
        enemigo_z = np.linspace(params.altura_enemigo, 0, len(t))
        antimisil_x = np.linspace(0, params.distancia_defensa, len(t))
        antimisil_y = np.linspace(0, 0, len(t))
        antimisil_z = np.linspace(0, params.altura_enemigo, len(t))
        return t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z

    @staticmethod
    def verificar_intercepcion(params: ParametrosSimulacion, t: np.ndarray, enemigo_x: np.ndarray, enemigo_y: np.ndarray, antimisil_x: np.ndarray, antimisil_y: np.ndarray) -> Tuple[bool, float, float, Optional[float], Optional[float]]:
        # Implementación simplificada
        intercepcion = False
        tiempo_intercepcion = 0.0
        distancia_minima = float('inf')
        interceptar_x = None
        interceptar_y = None

        for i in range(len(t)):
            distancia = math.sqrt((enemigo_x[i] - antimisil_x[i])**2 + (enemigo_y[i] - antimisil_y[i])**2)
            if distancia < params.constantes.MARGEN_ERROR:
                intercepcion = True
                tiempo_intercepcion = t[i]
                interceptar_x = enemigo_x[i]
                interceptar_y = enemigo_y[i]
                break
            if distancia < distancia_minima:
                distancia_minima = distancia

        return intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y

    @staticmethod
    def verificar_intercepcion_3d(params: ParametrosSimulacion, t: np.ndarray, enemigo_x: np.ndarray, enemigo_y: np.ndarray, enemigo_z: np.ndarray, antimisil_x: np.ndarray, antimisil_y: np.ndarray, antimisil_z: np.ndarray) -> Tuple[bool, float, float, Optional[float], Optional[float], Optional[float]]:
        # Implementación simplificada
        intercepcion = False
        tiempo_intercepcion = 0.0
        distancia_minima = float('inf')
        interceptar_x = None
        interceptar_y = None
        interceptar_z = None

        for i in range(len(t)):
            distancia = math.sqrt((enemigo_x[i] - antimisil_x[i])**2 + (enemigo_y[i] - antimisil_y[i])**2 + (enemigo_z[i] - antimisil_z[i])**2)
            if distancia < params.constantes.MARGEN_ERROR:
                intercepcion = True
                tiempo_intercepcion = t[i]
                interceptar_x = enemigo_x[i]
                interceptar_y = enemigo_y[i]
                interceptar_z = enemigo_z[i]
                break
            if distancia < distancia_minima:
                distancia_minima = distancia

        return intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y, interceptar_z

    @staticmethod
    def calcular_parametros_optimos(params: ParametrosSimulacion) -> Tuple[float, float]:
        # Implementación simplificada
        angulo_optimo = 45.0
        tiempo_optimo = 5.0
        return angulo_optimo, tiempo_optimo

class MotorSimulacion:
    pass

# Resto del código...
class GestorTema:
    """Gestor del tema de la aplicación"""

    def __init__(self, tema_inicial: Tema = Tema.OSCURO):
        self.tema_actual = tema_inicial
        self._aplicar_tema()

    def cambiar_tema(self) -> Tema:
        """Cambia entre tema claro y oscuro"""
        self.tema_actual = Tema.CLARO if self.tema_actual == Tema.OSCURO else Tema.OSCURO
        self._aplicar_tema()
        return self.tema_actual

    def _aplicar_tema(self) -> None:
        """Aplica el tema actual a la interfaz"""
        nombre_tema = "dark" if self.tema_actual == Tema.OSCURO else "light"
        sv_ttk.set_theme(nombre_tema)

    @property
    def es_oscuro(self) -> bool:
        """Devuelve True si el tema actual es oscuro"""
        return self.tema_actual == Tema.OSCURO

class GestorGrafico:
    """Clase para gestionar la visualización de gráficos"""

    def __init__(self, fig: Figure, ax: plt.Axes, canvas: FigureCanvasTkAgg, gestor_tema: GestorTema):
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.gestor_tema = gestor_tema
        self._actualizar_tema()

    def _actualizar_tema(self) -> None:
        """Actualiza el tema del gráfico"""
        es_oscuro = self.gestor_tema.es_oscuro

        # Configurar colores según el tema
        color_fondo = '#2e2e2e' if es_oscuro else 'white'
        color_fondo_grafico = '#3e3e3e' if es_oscuro else '#f0f0f0'

        self.fig.set_facecolor(color_fondo)
        self.ax.set_facecolor(color_fondo_grafico)
        self.canvas.draw()

    @property
    def color_texto(self) -> str:
        """Devuelve el color del texto según el tema"""
        return 'white' if self.gestor_tema.es_oscuro else 'black'

    @property
    def color_cuadricula(self) -> str:
        """Devuelve el color de la cuadrícula según el tema"""
        return '#555555' if self.gestor_tema.es_oscuro else '#cccccc'

    @property
    def color_leyenda(self) -> str:
        """Devuelve el color de fondo de la leyenda según el tema"""
        return '#333333' if self.gestor_tema.es_oscuro else '#f0f0f0'

    def al_cambiar_tema(self) -> None:
        """Manejador para cambios de tema"""
        self._actualizar_tema()

    def graficar_escenario_inicial(self, params: ParametrosSimulacion) -> None:
        """Dibuja el escenario inicial sin trayectorias"""
        self.ax.clear()

        # Posición del misil enemigo (punto C)
        self.ax.plot(params.distancia_defensa, params.altura_enemigo, 'ro', markersize=10, label='Misil Enemigo (C)')

        # Posición del sistema de defensa (punto A)
        self.ax.plot(0, 0, 'bo', markersize=10, label='Sistema Antiaéreo (A)')

        # Posición del objetivo (punto B)
        self.ax.plot(params.distancia_defensa, 0, 'go', markersize=10, label='Objetivo (B)')

        # Dibujar línea discontinua que muestra la trayectoria
        self.ax.plot([params.distancia_defensa, params.distancia_defensa], [0, params.altura_enemigo], 'r--', alpha=0.5)

        self._configurar_estilo_grafico(params)
        self.canvas.draw()

    def graficar_simulacion_completa(self, params: ParametrosSimulacion,
                                      t: np.ndarray, enemigo_x: np.ndarray, enemigo_y: np.ndarray,
                                      antimisil_x: np.ndarray, antimisil_y: np.ndarray,
                                      intercepcion: bool = False, interceptar_x: Optional[float] = None,
                                      interceptar_y: Optional[float] = None) -> None:
        """Dibuja la simulación completa con trayectorias"""
        self.ax.clear()

        # Dibujar trayectorias con estilos mejorados
        self.ax.plot(enemigo_x, enemigo_y, 'r-', label='Trayectoria Misil Enemigo', linewidth=2)
        self.ax.plot(antimisil_x, antimisil_y, 'b-', label='Trayectoria Misil Antiaéreo', linewidth=2)

        # Dibujar posiciones
        self.ax.plot(0, 0, 'bo', markersize=10, label='Sistema Antiaéreo (A)')
        self.ax.plot(params.distancia_defensa, 0, 'go', markersize=10, label='Objetivo (B)')
        self.ax.plot(params.distancia_defensa, params.altura_enemigo, 'ro', markersize=10, label='Posición Inicial Misil Enemigo (C)')

        # Marcar punto de intercepción si existe
        if intercepcion and interceptar_x is not None and interceptar_y is not None:
            self.ax.plot(interceptar_x, interceptar_y, 'mo', markersize=12, label='Punto de Intercepción')
            self.ax.add_patch(plt.Circle((interceptar_x, interceptar_y), params.constantes.MARGEN_ERROR, fill=False,
                                       color='m', linestyle='--', alpha=0.7))

        self._configurar_estilo_grafico(params, max_x=np.max(antimisil_x), max_y=np.max(antimisil_y))
        self.canvas.draw()

    def preparar_animacion(self, params: ParametrosSimulacion) -> Tuple[plt.Line2D, plt.Line2D]:
        """Prepara el gráfico para animación"""
        self.ax.clear()

        # Dibujar posiciones iniciales
        self.ax.plot(0, 0, 'bo', markersize=10, label='Sistema Antiaéreo (A)')
        self.ax.plot(params.distancia_defensa, 0, 'go', markersize=10, label='Objetivo (B)')
        self.ax.plot(params.distancia_defensa, params.altura_enemigo, 'ro', markersize=10, label='Posición Inicial Misil Enemigo (C)')

        self._configurar_estilo_grafico(params)

        # Crear líneas vacías para las trayectorias con estilos mejorados
        trayectoria_enemigo, = self.ax.plot([], [], 'r-', linewidth=1.5, alpha=0.7)
        trayectoria_antimisil, = self.ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7)

        self.canvas.draw()
        return trayectoria_enemigo, trayectoria_antimisil

    def graficar_simulacion_3d(self, params: ParametrosSimulacion,
                               t: np.ndarray, enemigo_x: np.ndarray, enemigo_y: np.ndarray, enemigo_z: np.ndarray,
                               antimisil_x: np.ndarray, antimisil_y: np.ndarray, antimisil_z: np.ndarray,
                               intercepcion: bool = False, interceptar_x: Optional[float] = None,
                               interceptar_y: Optional[float] = None, interceptar_z: Optional[float] = None) -> None:
        """Dibuja la simulación completa con trayectorias en 3D"""
        self.ax.clear()

        # Convertir a 3D si no lo es ya
        if not isinstance(self.ax, Axes3D):
            self.ax = self.fig.add_subplot(111, projection='3d')

        # Dibujar trayectorias con estilos mejorados
        self.ax.plot(enemigo_x, enemigo_y, enemigo_z, 'r-', label='Trayectoria Misil Enemigo', linewidth=2)
        self.ax.plot(antimisil_x, antimisil_y, antimisil_z, 'b-', label='Trayectoria Misil Antiaéreo', linewidth=2)

        # Dibujar posiciones
        self.ax.plot([0], [0], [0], 'bo', markersize=10, label='Sistema Antiaéreo (A)')
        self.ax.plot([params.distancia_defensa], [0], [0], 'go', markersize=10, label='Objetivo (B)')
        self.ax.plot([params.distancia_defensa], [0], [params.altura_enemigo], 'ro', markersize=10, label='Posición Inicial Misil Enemigo (C)')

        # Marcar punto de intercepción si existe
        if intercepcion and interceptar_x is not None and interceptar_y is not None and interceptar_z is not None:
            self.ax.plot([interceptar_x], [interceptar_y], [interceptar_z], 'mo', markersize=12, label='Punto de Intercepción')

        self._configurar_estilo_grafico_3d(params, max_x=np.max(antimisil_x), max_y=np.max(antimisil_y), max_z=np.max(antimisil_z))
        self.canvas.draw()

    def _configurar_estilo_grafico(self, params: ParametrosSimulacion, max_x: Optional[float] = None, max_y: Optional[float] = None) -> None:
        """Configura el estilo del gráfico 2D"""
        # Configurar límites de los ejes con margen
        x_max = max(params.distancia_defensa * 1.1, max_x * 1.1 if max_x is not None else 0, 60000)
        y_max = max(params.altura_enemigo * 1.1, max_y * 1.1 if max_y is not None else 0, 15000)

        self.ax.set_xlim([-5000, x_max])
        self.ax.set_ylim([-500, y_max])

        # Etiquetas y cuadrícula con estilos personalizados
        self.ax.set_xlabel('Distancia Horizontal (m)', color=self.color_texto, fontsize=10)
        self.ax.set_ylabel('Altura (m)', color=self.color_texto, fontsize=10)
        self.ax.set_title('Simulación de Interceptación de Misiles', color=self.color_texto, fontsize=14, fontweight='bold')
        self.ax.grid(True, color=self.color_cuadricula, linestyle='--', alpha=0.7)

        # Actualizar colores de las marcas
        self.ax.tick_params(axis='x', colors=self.color_texto)
        self.ax.tick_params(axis='y', colors=self.color_texto)

        # Leyenda con fondo personalizado y estilo moderno
        leyenda = self.ax.legend(facecolor=self.color_leyenda, framealpha=0.9)
        for texto in leyenda.get_texts():
            texto.set_color(self.color_texto)

        # Convertir ejes a km para mejor legibilidad
        def formato_coord(x: float, y: float) -> str:
            return f'x={x/1000:.2f} km, y={y/1000:.2f} km'

        self.ax.format_coord = formato_coord

    def _configurar_estilo_grafico_3d(self, params: ParametrosSimulacion, max_x: Optional[float] = None, max_y: Optional[float] = None, max_z: Optional[float] = None) -> None:
        """Configura el estilo del gráfico 3D"""
        # Configurar límites de los ejes con margen
        x_max = max(params.distancia_defensa * 1.1, max_x * 1.1 if max_x is not None else 0, 60000)
        y_max = max(params.altura_enemigo * 1.1, max_y * 1.1 if max_y is not None else 0, 15000)
        z_max = max(params.altura_enemigo * 1.1, max_z * 1.1 if max_z is not None else 0, 15000)

        self.ax.set_xlim([-5000, x_max])
        self.ax.set_ylim([-5000, y_max])
        self.ax.set_zlim([-500, z_max])

        # Etiquetas y cuadrícula con estilos personalizados
        self.ax.set_xlabel('Distancia X (m)', color=self.color_texto, fontsize=10)
        self.ax.set_ylabel('Distancia Y (m)', color=self.color_texto, fontsize=10)
        self.ax.set_zlabel('Altura (m)', color=self.color_texto, fontsize=10)
        self.ax.set_title('Simulación 3D de Interceptación de Misiles', color=self.color_texto, fontsize=14, fontweight='bold')
        self.ax.grid(True, color=self.color_cuadricula, linestyle='--', alpha=0.7)

        # Actualizar colores de las marcas
        self.ax.tick_params(axis='x', colors=self.color_texto)
        self.ax.tick_params(axis='y', colors=self.color_texto)
        self.ax.tick_params(axis='z', colors=self.color_texto)

        # Leyenda con fondo personalizado y estilo moderno
        leyenda = self.ax.legend(facecolor=self.color_leyenda, framealpha=0.9)
        for texto in leyenda.get_texts():
            texto.set_color(self.color_texto)

        # Convertir ejes a km para mejor legibilidad
        def formato_coord(x: float, y: float, z: float) -> str:
            return f'x={x/1000:.2f} km, y={y/1000:.2f} km, z={z/1000:.2f} km'

        self.ax.format_coord = formato_coord

class GestorPreajustes:
    """Gestor de escenarios predefinidos"""

    # Constantes para los presets
    ESCENARIO_ESTANDAR: Final[Dict[str, float]] = {"distancia_defensa": 30000, "altura_enemigo": 10000}
    DEFENSA_CERCANA: Final[Dict[str, float]] = {"distancia_defensa": 15000, "altura_enemigo": 10000}
    DEFENSA_LEJANA: Final[Dict[str, float]] = {"distancia_defensa": 50000, "altura_enemigo": 10000}
    GRAN_ALTITUD: Final[Dict[str, float]] = {"distancia_defensa": 30000, "altura_enemigo": 15000}

    @classmethod
    def aplicar_preajuste(cls, params: ParametrosSimulacion, nombre_preajuste: str) -> None:
        """Aplica un preset a los parámetros de simulación"""
        preajustes = {
            "estandar": cls.ESCENARIO_ESTANDAR,
            "cercana": cls.DEFENSA_CERCANA,
            "lejana": cls.DEFENSA_LEJANA,
            "alta": cls.GRAN_ALTITUD
        }

        if nombre_preajuste in preajustes:
            for clave, valor in preajustes[nombre_preajuste].items():
                setattr(params, clave, valor)

class GestorUI:
    """Gestor para componentes de la interfaz de usuario"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.estilo = ttk.Style()
        self._configurar_estilos()

    def _configurar_estilos(self) -> None:
        """Configura estilos personalizados para widgets"""
        # Personalizar etiquetas
        self.estilo.configure('Titulo.TLabel', font=('Segoe UI', 14, 'bold'))
        self.estilo.configure('Subtitulo.TLabel', font=('Segoe UI', 12))
        self.estilo.configure('Informacion.TLabel', font=('Segoe UI', 10))
        self.estilo.configure('Estado.TLabel', font=('Segoe UI', 10, 'italic'))

        # Personalizar botones
        self.estilo.configure('Primario.TButton', font=('Segoe UI', 11))
        self.estilo.configure('Secundario.TButton', font=('Segoe UI', 10))
        self.estilo.configure('Peligro.TButton', font=('Segoe UI', 10), foreground='red')

        # Personalizar frames
        self.estilo.configure('Tarjeta.TFrame', relief='raised')

    def crear_tooltip(self, widget: tk.Widget, texto: str) -> None:
        """Crea un tooltip para un widget"""

        def entrar(evento):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            # Crear ventana emergente
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")

            etiqueta = ttk.Label(self.tooltip, text=texto, justify=tk.LEFT,
                               background="#ffffe0", relief="solid", borderwidth=1,
                               font=("Segoe UI", 9, "normal"))
            etiqueta.pack(ipadx=3, ipady=2)

        def salir(evento):
            if hasattr(self, "tooltip"):
                self.tooltip.destroy()

        widget.bind("<Enter>", entrar)
        widget.bind("<Leave>", salir)

class ControladorAnimacion:
    """Controlador para animaciones de la simulación"""

    def __init__(self):
        self.en_ejecucion = False
        self.hilo: Optional[threading.Thread] = None
        self.evento_detener = threading.Event()

    def iniciar_animacion(self, funcion_animacion: Callable[[], None]) -> None:
        """Inicia una animación en un hilo separado"""
        if self.en_ejecucion:
            return

        self.evento_detener.clear()
        self.en_ejecucion = True
        self.hilo = threading.Thread(target=self._ejecutar_animacion, args=(funcion_animacion,))
        self.hilo.daemon = True  # El hilo se cerrará cuando el programa principal termine
        self.hilo.start()

    def _ejecutar_animacion(self, funcion_animacion: Callable[[], None]) -> None:
        """Ejecuta la función de animación y maneja la finalización"""
        try:
            funcion_animacion()
        finally:
            self.en_ejecucion = False

    def detener_animacion(self) -> None:
        """Detiene la animación en curso"""
        if not self.en_ejecucion:
            return

        self.evento_detener.set()
        if self.hilo and self.hilo.is_alive():
            self.hilo.join(timeout=1.0)

    @property
    def esta_ejecutando(self) -> bool:
        """Devuelve si la animación está en ejecución"""
        return self.en_ejecucion

class FormateadorResultados:
    """Clase para formatear los resultados de la simulación"""

    @staticmethod
    def formatear_resultados_intercepcion(intercepcion: bool, tiempo_intercepcion: float, distancia_minima: float,
                                           interceptar_x: Optional[float], interceptar_y: Optional[float],
                                           params: ParametrosSimulacion) -> Tuple[str, str]:
        """Formatea los resultados de la intercepción para mostrar en la interfaz"""
        texto_resultado = "Resultados de la Simulación:\n\n"
        texto_resultado += f"Intercepción exitosa: {'Sí' if intercepcion else 'No'}\n"
        texto_resultado += f"Tiempo de intercepción: {tiempo_intercepcion:.2f} segundos\n"
        texto_resultado += f"Distancia mínima: {distancia_minima:.2f} metros\n"

        if intercepcion and interceptar_x is not None and interceptar_y is not None:
            texto_resultado += f"Coordenadas de intercepción: ({interceptar_x:.2f}, {interceptar_y:.2f})\n"
        else:
            texto_resultado += "No se produjo intercepción.\n"

        texto_estado = "Simulación completada con éxito." if intercepcion else "Simulación completada sin intercepción."

        return texto_resultado, texto_estado

class AplicacionIntercepcionMisiles:
    """Aplicación principal para la simulación de intercepción de misiles"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simulación Avanzada de Intercepción de Misiles")
        self.root.geometry("1280x800")
        self.root.protocol("WM_DELETE_WINDOW", self.al_cerrar)

        # Inicializar componentes principales
        self.params = ParametrosSimulacion()
        self.gestor_tema = GestorTema()
        self.gestor_ui = GestorUI(root)
        self.controlador_animacion = ControladorAnimacion()
        self.estado_simulacion = EstadoSimulacion.INICIAL

        # Crear variables de control para la interfaz
        self._crear_variables_control()

        # Crear la GUI
        self._crear_widgets()

        # Inicializar gráfico
        self._configurar_grafico()

        # Inicializar variables de la GUI con valores predeterminados
        self._actualizar_gui_desde_parametros()

        # Gráfico inicial
        self.gestor_grafico.graficar_escenario_inicial(self.params)

        # Configurar temporizador para actualización periódica (para animaciones fluidas)
        self.root.after(100, self._actualizacion_periodica)

    def _crear_variables_control(self) -> None:
        """Crea las variables de control para la interfaz"""
        self.altura_enemigo_var = tk.DoubleVar()
        self.distancia_defensa_var = tk.DoubleVar()
        self.velocidad_antimisil_var = tk.DoubleVar()
        self.angulo_antimisil_var = tk.DoubleVar()
        self.tiempo_antimisil_var = tk.DoubleVar()
        self.velocidad_animacion_var = tk.DoubleVar()
        self.resultados_var = tk.StringVar(value="Resultados de la simulación aparecerán aquí.")
        self.estado_var = tk.StringVar(value="Sistema listo para simulación.")
        self.modo_vista_var = tk.StringVar(value="2D")  # Nueva variable para el modo de visualización

    def _configurar_grafico(self) -> None:
        """Configura el área de gráficos"""
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Crear canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.marco_grafico)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicializar el gestor de gráficos
        self.gestor_grafico = GestorGrafico(self.fig, self.ax, self.canvas, self.gestor_tema)

    def _crear_widgets(self) -> None:
        """Crea todos los widgets para la interfaz gráfica"""
        # Configuración básica
        self.marco_principal = ttk.Frame(self.root, padding="10")
        self.marco_principal.pack(fill=tk.BOTH, expand=True)

        # Crear la barra superior
        self._crear_barra_superior()

        # Crear panel de control
        self._crear_panel_control()

        # Crear panel de visualización
        self._crear_panel_visualizacion()

        # Crear barra de estado
        self._crear_barra_estado()

    def _crear_barra_superior(self) -> None:
        """Crea la barra superior con título y controles globales"""
        marco_titulo = ttk.Frame(self.marco_principal)
        marco_titulo.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            marco_titulo,
            text="Simulación Avanzada de Intercepción de Misiles",
            style='Titulo.TLabel'
        ).pack(side=tk.LEFT)

        # Frame para los botones de tema y modo de visualización
        marco_botones = ttk.Frame(marco_titulo)
        marco_botones.pack(side=tk.RIGHT)

        # Botón de cambio de tema
        boton_tema = ttk.Button(
            marco_botones,
            text="Cambiar Tema",
            command=self.cambiar_tema
        )
        boton_tema.pack(side=tk.LEFT, padx=5)

        # Botón para cambiar el modo de visualización
        self.boton_modo_vista = ttk.Button(
            marco_botones,
            text="Cambiar a 3D",
            command=self.cambiar_modo_vista
        )
        self.boton_modo_vista.pack(side=tk.LEFT, padx=5)

        # Botón para zoom in
        boton_zoom_in = ttk.Button(
            marco_botones,
            text="Acercar",
            command=self.zoom_in
        )
        boton_zoom_in.pack(side=tk.LEFT, padx=5)

        # Botón para zoom out
        boton_zoom_out = ttk.Button(
            marco_botones,
            text="Alejar",
            command=self.zoom_out
        )
        boton_zoom_out.pack(side=tk.LEFT, padx=5)

        # Botón para reset zoom
        boton_reset_zoom = ttk.Button(
            marco_botones,
            text="Restablecer Zoom",
            command=self.reset_zoom
        )
        boton_reset_zoom.pack(side=tk.LEFT, padx=5)

    def _crear_panel_control(self) -> None:
        """Crea el panel de control con todos los parámetros"""
        marco_control = ttk.LabelFrame(self.marco_principal, text="Panel de Control", padding="15")
        marco_control.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=10)

        # Parámetros del misil enemigo
        self._crear_controles_enemigo(marco_control)

        # Parámetros del sistema de defensa
        self._crear_controles_defensa(marco_control)

        # Parámetros de misil antiaéreo
        self._crear_controles_antimisil(marco_control)

        # Tiempo de lanzamiento
        self._crear_controles_tiempo(marco_control)

        # Velocidad de animación
        self._crear_controles_animacion(marco_control)

        # Botones de acción
        self._crear_botones_accion(marco_control)

    def _crear_controles_enemigo(self, padre: ttk.Frame) -> None:
        """Crea los controles para los parámetros del misil enemigo"""
        marco_enemigo = ttk.LabelFrame(padre, text="Parámetros del Misil Enemigo", style='Tarjeta.TFrame')
        marco_enemigo.pack(fill=tk.X, pady=5)

        ttk.Label(marco_enemigo, text="Altura del Misil Enemigo (m):", style='Subtitulo.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(marco_enemigo, textvariable=self.altura_enemigo_var).grid(row=0, column=1, padx=5, pady=2)

    def _crear_controles_defensa(self, padre: ttk.Frame) -> None:
        """Crea los controles para los parámetros del sistema de defensa"""
        marco_defensa = ttk.LabelFrame(padre, text="Parámetros del Sistema de Defensa", style='Tarjeta.TFrame')
        marco_defensa.pack(fill=tk.X, pady=5)

        ttk.Label(marco_defensa, text="Distancia al Objetivo (m):", style='Subtitulo.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(marco_defensa, textvariable=self.distancia_defensa_var).grid(row=0, column=1, padx=5, pady=2)

    def _crear_controles_antimisil(self, padre: ttk.Frame) -> None:
        """Crea los controles para los parámetros del misil antiaéreo"""
        marco_antimisil = ttk.LabelFrame(padre, text="Parámetros del Misil Antiaéreo", style='Tarjeta.TFrame')
        marco_antimisil.pack(fill=tk.X, pady=5)

        ttk.Label(marco_antimisil, text="Velocidad del Misil Antiaéreo (m/s):", style='Subtitulo.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(marco_antimisil, textvariable=self.velocidad_antimisil_var).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(marco_antimisil, text="Ángulo de Lanzamiento (grados):", style='Subtitulo.TLabel').grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(marco_antimisil, textvariable=self.angulo_antimisil_var).grid(row=1, column=1, padx=5, pady=2)

    def _crear_controles_tiempo(self, padre: ttk.Frame) -> None:
        """Crea los controles para el tiempo de lanzamiento"""
        marco_tiempo = ttk.LabelFrame(padre, text="Tiempo de Lanzamiento", style='Tarjeta.TFrame')
        marco_tiempo.pack(fill=tk.X, pady=5)

        ttk.Label(marco_tiempo, text="Tiempo de Lanzamiento (s):", style='Subtitulo.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(marco_tiempo, textvariable=self.tiempo_antimisil_var).grid(row=0, column=1, padx=5, pady=2)

    def _crear_controles_animacion(self, padre: ttk.Frame) -> None:
        """Crea los controles para la velocidad de animación"""
        marco_animacion = ttk.LabelFrame(padre, text="Velocidad de Animación", style='Tarjeta.TFrame')
        marco_animacion.pack(fill=tk.X, pady=5)

        ttk.Label(marco_animacion, text="Multiplicador de Velocidad:", style='Subtitulo.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(marco_animacion, textvariable=self.velocidad_animacion_var).grid(row=0, column=1, padx=5, pady=2)

    def _crear_botones_accion(self, padre: ttk.Frame) -> None:
        """Crea los botones de acción"""
        marco_accion = ttk.Frame(padre)
        marco_accion.pack(fill=tk.X, pady=10)

        ttk.Button(marco_accion, text="Ejecutar Simulación", command=self.ejecutar_simulacion, style='Primario.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(marco_accion, text="Animar Simulación", command=self.animar_simulacion, style='Primario.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(marco_accion, text="Calcular Parámetros Óptimos", command=self.calcular_parametros_optimos, style='Secundario.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(marco_accion, text="Reiniciar Simulación", command=self.reiniciar_simulacion, style='Peligro.TButton').pack(fill=tk.X, pady=2)

    def _crear_panel_visualizacion(self) -> None:
        """Crea el panel de visualización para el gráfico"""
        self.marco_grafico = ttk.LabelFrame(self.marco_principal, text="Visualización de la Simulación", padding="10")
        self.marco_grafico.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _crear_barra_estado(self) -> None:
        """Crea la barra de estado en la parte inferior"""
        marco_estado = ttk.Frame(self.marco_principal)
        marco_estado.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        ttk.Label(marco_estado, textvariable=self.estado_var, style='Estado.TLabel').pack(side=tk.LEFT, padx=5)

        self.texto_resultados = tk.Text(marco_estado, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.texto_resultados.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def _actualizar_gui_desde_parametros(self) -> None:
        """Actualiza la interfaz gráfica con los valores de los parámetros"""
        self.altura_enemigo_var.set(self.params.altura_enemigo)
        self.distancia_defensa_var.set(self.params.distancia_defensa)
        self.velocidad_antimisil_var.set(self.params.velocidad_antimisil)
        self.angulo_antimisil_var.set(self.params.angulo_antimisil)
        self.tiempo_antimisil_var.set(self.params.tiempo_inicio_antimisil)
        self.velocidad_animacion_var.set(self.params.velocidad_animacion)
        self.modo_vista_var.set("3D" if self.params.modo_vista == ModoVista.MODO_3D else "2D")

    def _actualizar_parametros_desde_gui(self) -> None:
        """Actualiza los parámetros de simulación desde la interfaz gráfica"""
        self.params.altura_enemigo = self.altura_enemigo_var.get()
        self.params.distancia_defensa = self.distancia_defensa_var.get()
        self.params.velocidad_antimisil = self.velocidad_antimisil_var.get()
        self.params.angulo_antimisil = self.angulo_antimisil_var.get()
        self.params.tiempo_inicio_antimisil = self.tiempo_antimisil_var.get()
        self.params.velocidad_animacion = self.velocidad_animacion_var.get()
        self.params.modo_vista = ModoVista.MODO_3D if self.modo_vista_var.get() == "3D" else ModoVista.MODO_2D

    def cambiar_tema(self) -> None:
        """Cambia el tema de la interfaz"""
        nuevo_tema = self.gestor_tema.cambiar_tema()
        self.gestor_grafico.al_cambiar_tema()
        self.estado_var.set(f"Tema cambiado a {'oscuro' if nuevo_tema == Tema.OSCURO else 'claro'}.")

    def cambiar_modo_vista(self) -> None:
        """Cambia entre los modos de visualización 2D y 3D"""
        if self.modo_vista_var.get() == "2D":
            self.modo_vista_var.set("3D")
            self.params.modo_vista = ModoVista.MODO_3D
            self.boton_modo_vista.config(text="Cambiar a 2D")
        else:
            self.modo_vista_var.set("2D")
            self.params.modo_vista = ModoVista.MODO_2D
            self.boton_modo_vista.config(text="Cambiar a 3D")

        self.estado_var.set(f"Modo de visualización cambiado a {self.modo_vista_var.get()}.")
        self.gestor_grafico.graficar_escenario_inicial(self.params)

    def zoom_in(self) -> None:
        """Zoom in on the graph"""
        self.ax.set_xlim([self.ax.get_xlim()[0] * 0.9, self.ax.get_xlim()[1] * 0.9])
        self.ax.set_ylim([self.ax.get_ylim()[0] * 0.9, self.ax.get_ylim()[1] * 0.9])
        if isinstance(self.ax, Axes3D):
            self.ax.set_zlim([self.ax.get_zlim()[0] * 0.9, self.ax.get_zlim()[1] * 0.9])
        self.canvas.draw()

    def zoom_out(self) -> None:
        """Zoom out on the graph"""
        self.ax.set_xlim([self.ax.get_xlim()[0] * 1.1, self.ax.get_xlim()[1] * 1.1])
        self.ax.set_ylim([self.ax.get_ylim()[0] * 1.1, self.ax.get_ylim()[1] * 1.1])
        if isinstance(self.ax, Axes3D):
            self.ax.set_zlim([self.ax.get_zlim()[0] * 1.1, self.ax.get_zlim()[1] * 1.1])
        self.canvas.draw()

    def reset_zoom(self) -> None:
        """Reset the zoom to default view"""
        self.gestor_grafico.graficar_escenario_inicial(self.params)

    def ejecutar_simulacion(self) -> None:
        """Ejecuta la simulación y muestra los resultados"""
        self._actualizar_parametros_desde_gui()
        if self.params.modo_vista == ModoVista.MODO_2D:
            t, enemigo_x, enemigo_y, antimisil_x, antimisil_y = CalculadorTrayectorias.calcular_trayectorias(self.params)
            intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y = CalculadorTrayectorias.verificar_intercepcion(
                self.params, t, enemigo_x, enemigo_y, antimisil_x, antimisil_y
            )
            self.gestor_grafico.graficar_simulacion_completa(self.params, t, enemigo_x, enemigo_y, antimisil_x, antimisil_y, intercepcion, interceptar_x, interceptar_y)
        else:
            t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z = CalculadorTrayectorias.calcular_trayectorias_3d(self.params)
            intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y, interceptar_z = CalculadorTrayectorias.verificar_intercepcion_3d(
                self.params, t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z
            )
            self.gestor_grafico.graficar_simulacion_3d(self.params, t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z, intercepcion, interceptar_x, interceptar_y, interceptar_z)

        texto_resultado, texto_estado = FormateadorResultados.formatear_resultados_intercepcion(
            intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y, self.params
        )

        self.texto_resultados.config(state=tk.NORMAL)
        self.texto_resultados.delete(1.0, tk.END)
        self.texto_resultados.insert(tk.END, texto_resultado)
        self.texto_resultados.config(state=tk.DISABLED)
        self.estado_var.set(texto_estado)
        self.estado_simulacion = EstadoSimulacion.COMPLETADO

    def animar_simulacion(self) -> None:
        """Anima la simulación paso a paso"""
        self._actualizar_parametros_desde_gui()
        self.estado_simulacion = EstadoSimulacion.ANIMANDO
        self.estado_var.set("Animación en progreso...")

        if self.params.modo_vista == ModoVista.MODO_2D:
            t, enemigo_x, enemigo_y, antimisil_x, antimisil_y = CalculadorTrayectorias.calcular_trayectorias(self.params)
            trayectoria_enemigo, trayectoria_antimisil = self.gestor_grafico.preparar_animacion(self.params)

            def animar():
                for i in range(len(t)):
                    if self.controlador_animacion.evento_detener.is_set():
                        break
                    trayectoria_enemigo.set_data(enemigo_x[:i], enemigo_y[:i])
                    trayectoria_antimisil.set_data(antimisil_x[:i], antimisil_y[:i])
                    self.canvas.draw()
                    time.sleep(self.params.constantes.PASO_TIEMPO / self.params.velocidad_animacion)

                intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y = CalculadorTrayectorias.verificar_intercepcion(
                    self.params, t, enemigo_x, enemigo_y, antimisil_x, antimisil_y
                )

                texto_resultado, texto_estado = FormateadorResultados.formatear_resultados_intercepcion(
                    intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y, self.params
                )

                self.texto_resultados.config(state=tk.NORMAL)
                self.texto_resultados.delete(1.0, tk.END)
                self.texto_resultados.insert(tk.END, texto_resultado)
                self.texto_resultados.config(state=tk.DISABLED)
                self.estado_var.set(texto_estado)

                self.gestor_grafico.graficar_simulacion_completa(self.params, t, enemigo_x, enemigo_y, antimisil_x, antimisil_y, intercepcion, interceptar_x, interceptar_y)
                self.estado_simulacion = EstadoSimulacion.COMPLETADO

        else:
            t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z = CalculadorTrayectorias.calcular_trayectorias_3d(self.params)
            trayectoria_enemigo, trayectoria_antimisil = self.gestor_grafico.preparar_animacion(self.params)

            def animar():
                for i in range(len(t)):
                    if self.controlador_animacion.evento_detener.is_set():
                        break
                    trayectoria_enemigo.set_data(enemigo_x[:i], enemigo_y[:i])
                    trayectoria_enemigo.set_3d_properties(enemigo_z[:i])
                    trayectoria_antimisil.set_data(antimisil_x[:i], antimisil_y[:i])
                    trayectoria_antimisil.set_3d_properties(antimisil_z[:i])
                    self.canvas.draw()
                    time.sleep(self.params.constantes.PASO_TIEMPO / self.params.velocidad_animacion)

                intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y, interceptar_z = CalculadorTrayectorias.verificar_intercepcion_3d(
                    self.params, t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z
                )

                texto_resultado, texto_estado = FormateadorResultados.formatear_resultados_intercepcion(
                    intercepcion, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y, self.params
                )

                self.texto_resultados.config(state=tk.NORMAL)
                self.texto_resultados.delete(1.0, tk.END)
                self.texto_resultados.insert(tk.END, texto_resultado)
                self.texto_resultados.config(state=tk.DISABLED)
                self.estado_var.set(texto_estado)

                self.gestor_grafico.graficar_simulacion_3d(self.params, t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z, intercepcion, interceptar_x, interceptar_y, interceptar_z)
                self.estado_simulacion = EstadoSimulacion.COMPLETADO

        self.controlador_animacion.iniciar_animacion(animar)

    def calcular_parametros_optimos(self) -> None:
        """Calcula y aplica los parámetros óptimos para la intercepción"""
        angulo_optimo, tiempo_optimo = CalculadorTrayectorias.calcular_parametros_optimos(self.params)
        self.params.angulo_antimisil = angulo_optimo
        self.params.tiempo_inicio_antimisil = tiempo_optimo
        self._actualizar_gui_desde_parametros()
        self.estado_var.set("Parámetros óptimos calculados y aplicados.")

    def reiniciar_simulacion(self) -> None:
        """Reinicia la simulación a los valores predeterminados"""
        self.params = ParametrosSimulacion()
        self._actualizar_gui_desde_parametros()
        self.gestor_grafico.graficar_escenario_inicial(self.params)
        self.texto_resultados.config(state=tk.NORMAL)
        self.texto_resultados.delete(1.0, tk.END)
        self.texto_resultados.config(state=tk.DISABLED)
        self.estado_var.set("Simulación reiniciada.")
        self.estado_simulacion = EstadoSimulacion.INICIAL

    def al_cerrar(self) -> None:
        """Manejador para el cierre de la ventana"""
        if self.controlador_animacion.esta_ejecutando:
            self.controlador_animacion.detener_animacion()
        self.root.destroy()

    def _actualizacion_periodica(self) -> None:
        """Actualización periódica para manejar eventos y animaciones"""
        if self.estado_simulacion == EstadoSimulacion.ANIMANDO and not self.controlador_animacion.esta_ejecutando:
            self.estado_simulacion = EstadoSimulacion.COMPLETADO
        self.root.after(100, self._actualizacion_periodica)

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionIntercepcionMisiles(root)
    root.mainloop()
