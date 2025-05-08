from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Callable, TypeAlias, Final
import math
import random
import datetime
import numpy as np
from config import *

@dataclass
class EspecificacionMisil:
    """Especificaciones técnicas de un misil"""
    nombre: str
    velocidad: float  # m/s
    altura_maxima: float  # metros
    alcance: float  # metros
    masa: float  # kg
    coeficiente_arrastre: float  # adimensional
    seccion_transversal: float  # m²
    capacidad_combustible: float  # kg
    consumo_combustible: float  # kg/s
    sistema_guiado: str  # tipo de sistema de guiado
    rendimiento_explosivo: float  # kg TNT equivalente
    color: str = "#FF0000"  # color para la visualización

    def calcular_arrastre(self, velocidad: float, altitud: float, constantes: ConstantesSimulacion) -> float:
        """Calcula la fuerza de arrastre en función de la velocidad y altitud"""
        # Densidad del aire decrece exponencialmente con la altitud
        rho = constantes.DENSIDAD_AIRE_NIVEL_MAR * math.exp(-altitud/10000)
        return 0.5 * rho * self.coeficiente_arrastre * self.seccion_transversal * velocidad**2

@dataclass
class EfectosClima:
    """Efectos climáticos que afectan la simulación"""
    condicion: CondicionClima = CondicionClima.DESPEJADO
    velocidad_viento: float = 0.0  # m/s
    direccion_viento: float = 0.0  # grados (0 = norte, 90 = este)
    factor_turbulencia: float = 0.0  # adimensional (0-1)
    visibilidad: float = 10000  # metros

    def aplicar_efecto_viento(self, velocidad_x: float, velocidad_y: float) -> Tuple[float, float]:
        """Aplica el efecto del viento a la velocidad del misil"""
        if self.condicion == CondicionClima.DESPEJADO:
            return velocidad_x, velocidad_y

        viento_rad = math.radians(self.direccion_viento)
        viento_x = self.velocidad_viento * math.sin(viento_rad)
        viento_y = self.velocidad_viento * math.cos(viento_rad)

        # Añadir turbulencia aleatoria si corresponde
        if self.condicion in [CondicionClima.VENTOSO, CondicionClima.TORMENTOSO]:
            turbulencia_x = random.uniform(-1, 1) * self.factor_turbulencia * self.velocidad_viento
            turbulencia_y = random.uniform(-1, 1) * self.factor_turbulencia * self.velocidad_viento
            viento_x += turbulencia_x
            viento_y += turbulencia_y

        return velocidad_x + viento_x, velocidad_y + viento_y

@dataclass
class ParametrosSimulacion:
    """Clase para almacenar los parámetros de la simulación"""
    # Parámetros básicos
    altura_enemigo: float = 10000  # metros (10 km)
    distancia_defensa: float = 30000  # metros (30 km)
    velocidad_antimisil: float = 1000  # m/s
    angulo_antimisil: float = 45  # grados
    tiempo_inicio_antimisil: float = 0  # segundos
    velocidad_animacion: float = 1.0  # multiplicador

    # Parámetros avanzados
    modo_simulacion: ModoSimulacion = ModoSimulacion.BASICO
    modo_vista: ModoVista = ModoVista.MODO_2D
    velocidad_enemigo: float = 800  # m/s
    angulo_enemigo: float = 270  # grados (90 = este, 270 = oeste)
    multiples_amenazas: bool = False
    num_amenazas: int = 1
    conteo_sistemas_defensa: int = 1

    # Efectos ambientales
    clima: EfectosClima = field(default_factory=EfectosClima)

    # Especificaciones de misiles
    misil_enemigo: EspecificacionMisil = field(default_factory=lambda: EspecificacionMisil(
        nombre="Enemigo Estándar",
        velocidad=800,
        altura_maxima=15000,
        alcance=40000,
        masa=1000,
        coeficiente_arrastre=0.1,
        seccion_transversal=0.5,
        capacidad_combustible=500,
        consumo_combustible=5,
        sistema_guiado="Balístico",
        rendimiento_explosivo=500,
        color="#FF0000"
    ))

    misil_defensa: EspecificacionMisil = field(default_factory=lambda: EspecificacionMisil(
        nombre="Defensa Estándar",
        velocidad=1000,
        altura_maxima=20000,
        alcance=50000,
        masa=800,
        coeficiente_arrastre=0.05,
        seccion_transversal=0.3,
        capacidad_combustible=400,
        consumo_combustible=8,
        sistema_guiado="Activo",
        rendimiento_explosivo=50,
        color="#0000FF"
    ))

    # Constantes de simulación
    constantes: ConstantesSimulacion = field(default_factory=ConstantesSimulacion)

    # Parámetros de dificultad (para modo entrenamiento)
    nivel_dificultad: NivelDificultad = NivelDificultad.PRINCIPIANTE
    limite_tiempo: Optional[float] = None
    puntuacion: int = 0

    @property
    def tiempo_caida_libre(self) -> float:
        """Calcula el tiempo de caída libre"""
        return math.sqrt(2 * self.altura_enemigo / self.constantes.GRAVEDAD)

    def a_diccionario(self) -> Dict[str, Any]:
        """Convierte los parámetros a un diccionario para serialización"""
        return {
            "altura_enemigo": self.altura_enemigo,
            "distancia_defensa": self.distancia_defensa,
            "velocidad_antimisil": self.velocidad_antimisil,
            "angulo_antimisil": self.angulo_antimisil,
            "tiempo_inicio_antimisil": self.tiempo_inicio_antimisil,
            "velocidad_animacion": self.velocidad_animacion,
            "velocidad_enemigo": self.velocidad_enemigo,
            "angulo_enemigo": self.angulo_enemigo,
            "multiples_amenazas": self.multiples_amenazas,
            "num_amenazas": self.num_amenazas,
            "conteo_sistemas_defensa": self.conteo_sistemas_defensa,
            "modo_simulacion": self.modo_simulacion.name,
            "modo_vista": self.modo_vista.name,
            "clima": {
                "condicion": self.clima.condicion.name,
                "velocidad_viento": self.clima.velocidad_viento,
                "direccion_viento": self.clima.direccion_viento,
                "factor_turbulencia": self.clima.factor_turbulencia,
                "visibilidad": self.clima.visibilidad
            },
            "misil_enemigo": {
                "nombre": self.misil_enemigo.nombre,
                "velocidad": self.misil_enemigo.velocidad,
                "color": self.misil_enemigo.color
            },
            "misil_defensa": {
                "nombre": self.misil_defensa.nombre,
                "velocidad": self.misil_defensa.velocidad,
                "color": self.misil_defensa.color
            },
            "nivel_dificultad": self.nivel_dificultad.name if self.limite_tiempo else None,
            "limite_tiempo": self.limite_tiempo
        }

    @classmethod
    def desde_diccionario(cls, data: Dict[str, Any]) -> 'ParametrosSimulacion':
        """Crea una instancia desde un diccionario"""
        params = cls(
            altura_enemigo=data.get("altura_enemigo", 10000),
            distancia_defensa=data.get("distancia_defensa", 30000),
            velocidad_antimisil=data.get("velocidad_antimisil", 1000),
            angulo_antimisil=data.get("angulo_antimisil", 45),
            tiempo_inicio_antimisil=data.get("tiempo_inicio_antimisil", 0),
            velocidad_animacion=data.get("velocidad_animacion", 1.0),
            velocidad_enemigo=data.get("velocidad_enemigo", 800),
            angulo_enemigo=data.get("angulo_enemigo", 270),
            multiples_amenazas=data.get("multiples_amenazas", False),
            num_amenazas=data.get("num_amenazas", 1),
            conteo_sistemas_defensa=data.get("conteo_sistemas_defensa", 1)
        )

        # Configurar modo de simulación
        if "modo_simulacion" in data:
            params.modo_simulacion = ModoSimulacion[data["modo_simulacion"]]

        # Configurar modo de visualización
        if "modo_vista" in data:
            params.modo_vista = ModoVista[data["modo_vista"]]

        # Configurar efectos climáticos
        if "clima" in data:
            clima_data = data["clima"]
            params.clima = EfectosClima(
                condicion=CondicionClima[clima_data.get("condicion", "DESPEJADO")],
                velocidad_viento=clima_data.get("velocidad_viento", 0.0),
                direccion_viento=clima_data.get("direccion_viento", 0.0),
                factor_turbulencia=clima_data.get("factor_turbulencia", 0.0),
                visibilidad=clima_data.get("visibilidad", 10000)
            )

        # Configurar misiles
        if "misil_enemigo" in data:
            enemigo_data = data["misil_enemigo"]
            params.misil_enemigo.nombre = enemigo_data.get("nombre", params.misil_enemigo.nombre)
            params.misil_enemigo.velocidad = enemigo_data.get("velocidad", params.misil_enemigo.velocidad)
            params.misil_enemigo.color = enemigo_data.get("color", params.misil_enemigo.color)

        if "misil_defensa" in data:
            defensa_data = data["misil_defensa"]
            params.misil_defensa.nombre = defensa_data.get("nombre", params.misil_defensa.nombre)
            params.misil_defensa.velocidad = defensa_data.get("velocidad", params.misil_defensa.velocidad)
            params.misil_defensa.color = defensa_data.get("color", params.misil_defensa.color)

        # Configurar nivel de dificultad
        if "nivel_dificultad" in data and data["nivel_dificultad"]:
            params.nivel_dificultad = NivelDificultad[data["nivel_dificultad"]]
            params.limite_tiempo = data.get("limite_tiempo")

        return params

@dataclass
class ResultadosSimulacion:
    """Resultados de una simulación de interceptación"""
    exito: bool
    tiempo_intercepcion: float
    distancia_minima: float
    coords_intercepcion: Optional[Tuple[float, float, Optional[float]]]
    coords_impacto_enemigo: Optional[Tuple[float, float, Optional[float]]]
    tiempo_impacto_enemigo: Optional[float]
    tiempo_respuesta_defensa: float
    longitud_trayectoria_misil_defensa: float
    longitud_trayectoria_misil_enemigo: float
    combustible_consumido: float
    puntuacion_eficiencia: float
    probabilidad_exito: float
    bajas_evitadas: Optional[int]
    dano_evitado: Optional[float]
    marca_tiempo: datetime.datetime = field(default_factory=datetime.datetime.now)

    @property
    def es_optimo(self) -> bool:
        """Determina si la intercepción fue óptima"""
        return self.exito and self.puntuacion_eficiencia > 85

    def a_diccionario(self) -> Dict[str, Any]:
        """Convierte los resultados a un diccionario para serialización"""
        return {
            "marca_tiempo": self.marca_tiempo.isoformat(),
            "exito": self.exito,
            "tiempo_intercepcion": float(self.tiempo_intercepcion),
            "distancia_minima": float(self.distancia_minima),
            "coords_intercepcion": [float(x) if x is not None else None for x in self.coords_intercepcion] if self.coords_intercepcion else None,
            "coords_impacto_enemigo": [float(x) if x is not None else None for x in self.coords_impacto_enemigo] if self.coords_impacto_enemigo else None,
            "tiempo_impacto_enemigo": float(self.tiempo_impacto_enemigo) if self.tiempo_impacto_enemigo is not None else None,
            "tiempo_respuesta_defensa": float(self.tiempo_respuesta_defensa),
            "longitud_trayectoria_misil_defensa": float(self.longitud_trayectoria_misil_defensa),
            "longitud_trayectoria_misil_enemigo": float(self.longitud_trayectoria_misil_enemigo),
            "combustible_consumido": float(self.combustible_consumido),
            "puntuacion_eficiencia": float(self.puntuacion_eficiencia),
            "probabilidad_exito": float(self.probabilidad_exito),
            "bajas_evitadas": int(self.bajas_evitadas) if self.bajas_evitadas is not None else None,
            "dano_evitado": float(self.dano_evitado) if self.dano_evitado is not None else None
        }
