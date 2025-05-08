from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Callable, TypeAlias, Final
import numpy as np  # Import numpy module

# Tipos personalizados para mejorar la legibilidad
Posicion: TypeAlias = Tuple[float, float]
Posicion3D: TypeAlias = Tuple[float, float, float]
SecuenciaTiempo: TypeAlias = np.ndarray
Trayectoria: TypeAlias = Tuple[SecuenciaTiempo, np.ndarray, np.ndarray]
Trayectoria3D: TypeAlias = Tuple[SecuenciaTiempo, np.ndarray, np.ndarray, np.ndarray]  # tiempo, x, y, z
ResultadoIntercepcion: TypeAlias = Tuple[bool, float, float, Optional[float], Optional[float]]
ResultadoIntercepcion3D: TypeAlias = Tuple[bool, float, float, Optional[float], Optional[float], Optional[float]]

class TipoNotificacion(Enum):
    """Tipos de notificaciones para el sistema de alertas"""
    INFO = auto()
    EXITO = auto()
    ADVERTENCIA = auto()
    ERROR = auto()

class Tema(Enum):
    """Enumeración para los temas de la interfaz"""
    OSCURO = auto()
    CLARO = auto()
    MILITAR = auto()
    ESPACIO = auto()
    NEON = auto()

class TipoMisil(Enum):
    """Enumeración para los tipos de misiles"""
    ENEMIGO = auto()
    DEFENSA = auto()
    INTERCEPTOR = auto()

class EstadoSimulacion(Enum):
    """Estados posibles de la simulación"""
    INICIAL = auto()
    EJECUTANDO = auto()
    COMPLETADO = auto()
    ANIMANDO = auto()
    PAUSADO = auto()
    FALLIDO = auto()

class ModoSimulacion(Enum):
    """Modos de simulación disponibles"""
    BASICO = auto()
    AVANZADO = auto()
    EXPERTO = auto()
    ENTRENAMIENTO = auto()

class NivelDificultad(Enum):
    """Niveles de dificultad para el modo entrenamiento"""
    PRINCIPIANTE = auto()
    INTERMEDIO = auto()
    PROFESIONAL = auto()
    EXPERTO = auto()

class CondicionClima(Enum):
    """Condiciones climáticas para efectos ambientales"""
    DESPEJADO = auto()
    VENTOSO = auto()
    LLUVIOSO = auto()
    TORMENTOSO = auto()

class ModoVista(Enum):
    """Modos de visualización disponibles"""
    MODO_2D = auto()
    MODO_3D = auto()

class FormatoExportacion(Enum):
    """Formatos de exportación disponibles"""
    PNG = auto()
    PDF = auto()
    CSV = auto()
    JSON = auto()

@dataclass(frozen=True)
class ConstantesSimulacion:
    """Constantes físicas para la simulación"""
    GRAVEDAD: Final[float] = 9.8  # m/s²
    MARGEN_ERROR: Final[float] = 100  # metros
    PASO_TIEMPO: Final[float] = 0.1  # segundos
    DENSIDAD_AIRE_NIVEL_MAR: Final[float] = 1.225  # kg/m³
    VELOCIDAD_SONIDO: Final[float] = 343  # m/s
    RADIO_TIERRA: Final[float] = 6371000  # metros
    FACTOR_CORIOLIS: Final[float] = 0.0001  # factor de corrección Coriolis
