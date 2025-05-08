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
from config import *
from models import *

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
