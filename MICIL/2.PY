import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import math
from dataclasses import dataclass, field
import threading
from typing import Tuple, List, Optional, Dict, Any, Protocol, Final, TypeAlias, Callable
from enum import Enum, auto
from pathlib import Path
import contextlib
import sv_ttk  # Biblioteca para tema oscuro moderno
import datetime  # Importar el módulo datetime
import random  # Importar el módulo random

# Tipos personalizados para mejorar la legibilidad
Position: TypeAlias = Tuple[float, float]
Position3D: TypeAlias = Tuple[float, float, float]
TimeSequence: TypeAlias = np.ndarray
Trajectory: TypeAlias = Tuple[TimeSequence, np.ndarray, np.ndarray]
Trajectory3D: TypeAlias = Tuple[TimeSequence, np.ndarray, np.ndarray, np.ndarray]  # tiempo, x, y, z
InterceptionResult: TypeAlias = Tuple[bool, float, float, Optional[float], Optional[float]]
InterceptionResult3D: TypeAlias = Tuple[bool, float, float, Optional[float], Optional[float], Optional[float]]

class NotificationType(Enum):
    """Tipos de notificaciones para el sistema de alertas"""
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()

class Theme(Enum):
    """Enumeración para los temas de la interfaz"""
    DARK = auto()
    LIGHT = auto()
    MILITARY = auto()
    SPACE = auto()
    NEON = auto()

class MissileType(Enum):
    """Enumeración para los tipos de misiles"""
    ENEMY = auto()
    DEFENSE = auto()
    INTERCEPTOR = auto()

class SimulationState(Enum):
    """Estados posibles de la simulación"""
    INITIAL = auto()
    RUNNING = auto()
    COMPLETED = auto()
    ANIMATING = auto()
    PAUSED = auto()
    FAILED = auto()

class SimulationMode(Enum):
    """Modos de simulación disponibles"""
    BASIC = auto()
    ADVANCED = auto()
    EXPERT = auto()
    TRAINING = auto()

class DifficultyLevel(Enum):
    """Niveles de dificultad para el modo entrenamiento"""
    NOVICE = auto()
    INTERMEDIATE = auto()
    PROFESSIONAL = auto()
    EXPERT = auto()

class WeatherCondition(Enum):
    """Condiciones climáticas para efectos ambientales"""
    CLEAR = auto()
    WINDY = auto()
    RAINY = auto()
    STORMY = auto()

class ViewMode(Enum):
    """Modos de visualización disponibles"""
    MODE_2D = auto()
    MODE_3D = auto()

class ExportFormat(Enum):
    """Formatos de exportación disponibles"""
    PNG = auto()
    PDF = auto()
    CSV = auto()
    JSON = auto()

@dataclass(frozen=True)
class SimulationConstants:
    """Constantes físicas para la simulación"""
    GRAVITY: Final[float] = 9.8  # m/s²
    ERROR_MARGIN: Final[float] = 100  # metros
    TIME_STEP: Final[float] = 0.1  # segundos
    AIR_DENSITY_SEA_LEVEL: Final[float] = 1.225  # kg/m³
    SOUND_SPEED: Final[float] = 343  # m/s
    EARTH_RADIUS: Final[float] = 6371000  # metros
    CORIOLIS_FACTOR: Final[float] = 0.0001  # factor de corrección Coriolis

@dataclass
class MissileSpecification:
    """Especificaciones técnicas de un misil"""
    name: str
    velocity: float  # m/s
    max_altitude: float  # metros
    range: float  # metros
    mass: float  # kg
    drag_coefficient: float  # adimensional
    cross_section: float  # m²
    fuel_capacity: float  # kg
    fuel_consumption: float  # kg/s
    guidance_system: str  # tipo de sistema de guiado
    explosive_yield: float  # kg TNT equivalente
    color: str = "#FF0000"  # color para la visualización

    def calculate_drag(self, velocity: float, altitude: float, constants: SimulationConstants) -> float:
        """Calcula la fuerza de arrastre en función de la velocidad y altitud"""
        # Densidad del aire decrece exponencialmente con la altitud
        rho = constants.AIR_DENSITY_SEA_LEVEL * math.exp(-altitude/10000)
        return 0.5 * rho * self.drag_coefficient * self.cross_section * velocity**2

@dataclass
class WeatherEffects:
    """Efectos climáticos que afectan la simulación"""
    condition: WeatherCondition = WeatherCondition.CLEAR
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # grados (0 = norte, 90 = este)
    turbulence_factor: float = 0.0  # adimensional (0-1)
    visibility: float = 10000  # metros

    def apply_wind_effect(self, velocity_x: float, velocity_y: float) -> Tuple[float, float]:
        """Aplica el efecto del viento a la velocidad del misil"""
        if self.condition == WeatherCondition.CLEAR:
            return velocity_x, velocity_y

        wind_rad = math.radians(self.wind_direction)
        wind_x = self.wind_speed * math.sin(wind_rad)
        wind_y = self.wind_speed * math.cos(wind_rad)

        # Añadir turbulencia aleatoria si corresponde
        if self.condition in [WeatherCondition.WINDY, WeatherCondition.STORMY]:
            turbulence_x = random.uniform(-1, 1) * self.turbulence_factor * self.wind_speed
            turbulence_y = random.uniform(-1, 1) * self.turbulence_factor * self.wind_speed
            wind_x += turbulence_x
            wind_y += turbulence_y

        return velocity_x + wind_x, velocity_y + wind_y

@dataclass
class SimulationParameters:
    """Clase para almacenar los parámetros de la simulación"""
    # Parámetros básicos
    enemy_height: float = 10000  # metros (10 km)
    defense_distance: float = 30000  # metros (30 km)
    antimissile_velocity: float = 1000  # m/s
    antimissile_angle: float = 45  # grados
    antimissile_time_start: float = 0  # segundos
    animation_speed: float = 1.0  # multiplicador

    # Parámetros avanzados
    simulation_mode: SimulationMode = SimulationMode.BASIC
    view_mode: ViewMode = ViewMode.MODE_2D
    enemy_velocity: float = 800  # m/s
    enemy_angle: float = 270  # grados (90 = este, 270 = oeste)
    multiple_threats: bool = False
    num_threats: int = 1
    defense_system_count: int = 1

    # Efectos ambientales
    weather: WeatherEffects = field(default_factory=WeatherEffects)

    # Especificaciones de misiles
    enemy_missile: MissileSpecification = field(default_factory=lambda: MissileSpecification(
        name="Enemigo Estándar",
        velocity=800,
        max_altitude=15000,
        range=40000,
        mass=1000,
        drag_coefficient=0.1,
        cross_section=0.5,
        fuel_capacity=500,
        fuel_consumption=5,
        guidance_system="Balístico",
        explosive_yield=500,
        color="#FF0000"
    ))

    defense_missile: MissileSpecification = field(default_factory=lambda: MissileSpecification(
        name="Defensa Estándar",
        velocity=1000,
        max_altitude=20000,
        range=50000,
        mass=800,
        drag_coefficient=0.05,
        cross_section=0.3,
        fuel_capacity=400,
        fuel_consumption=8,
        guidance_system="Activo",
        explosive_yield=50,
        color="#0000FF"
    ))

    # Constantes de simulación
    constants: SimulationConstants = field(default_factory=SimulationConstants)

    # Parámetros de dificultad (para modo entrenamiento)
    difficulty_level: DifficultyLevel = DifficultyLevel.NOVICE
    time_limit: Optional[float] = None
    score: int = 0

    @property
    def free_fall_time(self) -> float:
        """Calcula el tiempo de caída libre"""
        return math.sqrt(2 * self.enemy_height / self.constants.GRAVITY)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte los parámetros a un diccionario para serialización"""
        return {
            "enemy_height": self.enemy_height,
            "defense_distance": self.defense_distance,
            "antimissile_velocity": self.antimissile_velocity,
            "antimissile_angle": self.antimissile_angle,
            "antimissile_time_start": self.antimissile_time_start,
            "animation_speed": self.animation_speed,
            "enemy_velocity": self.enemy_velocity,
            "enemy_angle": self.enemy_angle,
            "multiple_threats": self.multiple_threats,
            "num_threats": self.num_threats,
            "defense_system_count": self.defense_system_count,
            "simulation_mode": self.simulation_mode.name,
            "view_mode": self.view_mode.name,
            "weather": {
                "condition": self.weather.condition.name,
                "wind_speed": self.weather.wind_speed,
                "wind_direction": self.weather.wind_direction,
                "turbulence_factor": self.weather.turbulence_factor,
                "visibility": self.weather.visibility
            },
            "enemy_missile": {
                "name": self.enemy_missile.name,
                "velocity": self.enemy_missile.velocity,
                "color": self.enemy_missile.color
            },
            "defense_missile": {
                "name": self.defense_missile.name,
                "velocity": self.defense_missile.velocity,
                "color": self.defense_missile.color
            },
            "difficulty_level": self.difficulty_level.name if self.time_limit else None,
            "time_limit": self.time_limit
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationParameters':
        """Crea una instancia desde un diccionario"""
        params = cls(
            enemy_height=data.get("enemy_height", 10000),
            defense_distance=data.get("defense_distance", 30000),
            antimissile_velocity=data.get("antimissile_velocity", 1000),
            antimissile_angle=data.get("antimissile_angle", 45),
            antimissile_time_start=data.get("antimissile_time_start", 0),
            animation_speed=data.get("animation_speed", 1.0),
            enemy_velocity=data.get("enemy_velocity", 800),
            enemy_angle=data.get("enemy_angle", 270),
            multiple_threats=data.get("multiple_threats", False),
            num_threats=data.get("num_threats", 1),
            defense_system_count=data.get("defense_system_count", 1)
        )

        # Configurar modo de simulación
        if "simulation_mode" in data:
            params.simulation_mode = SimulationMode[data["simulation_mode"]]

        # Configurar modo de visualización
        if "view_mode" in data:
            params.view_mode = ViewMode[data["view_mode"]]

        # Configurar efectos climáticos
        if "weather" in data:
            weather_data = data["weather"]
            params.weather = WeatherEffects(
                condition=WeatherCondition[weather_data.get("condition", "CLEAR")],
                wind_speed=weather_data.get("wind_speed", 0.0),
                wind_direction=weather_data.get("wind_direction", 0.0),
                turbulence_factor=weather_data.get("turbulence_factor", 0.0),
                visibility=weather_data.get("visibility", 10000)
            )

        # Configurar misiles
        if "enemy_missile" in data:
            enemy_data = data["enemy_missile"]
            params.enemy_missile.name = enemy_data.get("name", params.enemy_missile.name)
            params.enemy_missile.velocity = enemy_data.get("velocity", params.enemy_missile.velocity)
            params.enemy_missile.color = enemy_data.get("color", params.enemy_missile.color)

        if "defense_missile" in data:
            defense_data = data["defense_missile"]
            params.defense_missile.name = defense_data.get("name", params.defense_missile.name)
            params.defense_missile.velocity = defense_data.get("velocity", params.defense_missile.velocity)
            params.defense_missile.color = defense_data.get("color", params.defense_missile.color)

        # Configurar nivel de dificultad
        if "difficulty_level" in data and data["difficulty_level"]:
            params.difficulty_level = DifficultyLevel[data["difficulty_level"]]
            params.time_limit = data.get("time_limit")

        return params

@dataclass
class SimulationResults:
    """Resultados de una simulación de interceptación"""
    success: bool
    intercept_time: float
    min_distance: float
    intercept_coords: Optional[Tuple[float, float, Optional[float]]]
    enemy_impact_coords: Optional[Tuple[float, float, Optional[float]]]
    enemy_time_to_impact: Optional[float]
    defense_response_time: float
    defense_missile_path_length: float
    enemy_missile_path_length: float
    fuel_consumed: float
    efficiency_score: float
    success_probability: float
    casualties_prevented: Optional[int]
    damage_prevented: Optional[float]
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

    @property
    def is_optimal(self) -> bool:
        """Determina si la intercepción fue óptima"""
        return self.success and self.efficiency_score > 85

    def to_dict(self) -> Dict[str, Any]:
        """Convierte los resultados a un diccionario para serialización"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "intercept_time": float(self.intercept_time),
            "min_distance": float(self.min_distance),
            "intercept_coords": [float(x) if x is not None else None for x in self.intercept_coords] if self.intercept_coords else None,
            "enemy_impact_coords": [float(x) if x is not None else None for x in self.enemy_impact_coords] if self.enemy_impact_coords else None,
            "enemy_time_to_impact": float(self.enemy_time_to_impact) if self.enemy_time_to_impact is not None else None,
            "defense_response_time": float(self.defense_response_time),
            "defense_missile_path_length": float(self.defense_missile_path_length),
            "enemy_missile_path_length": float(self.enemy_missile_path_length),
            "fuel_consumed": float(self.fuel_consumed),
            "efficiency_score": float(self.efficiency_score),
            "success_probability": float(self.success_probability),
            "casualties_prevented": int(self.casualties_prevented) if self.casualties_prevented is not None else None,
            "damage_prevented": float(self.damage_prevented) if self.damage_prevented is not None else None
        }

class TrajectoryCalculator:
    """Clase para calcular trayectorias de misiles"""

    @staticmethod
    def calculate_trajectories(params: SimulationParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcula las trayectorias para ambos misiles en 2D"""
        # Tiempo para toda la simulación (de 0 a poco después del tiempo de caída libre)
        max_time = params.free_fall_time * 1.5
        t = np.arange(0, max_time, params.constants.TIME_STEP)

        # Convertir ángulos a radianes
        enemy_angle_rad = np.radians(params.enemy_angle)
        antimissile_angle_rad = np.radians(params.antimissile_angle)

        # Trayectoria del misil enemigo (considerando componentes horizontal y vertical)
        enemy_v0x = params.enemy_velocity * np.cos(enemy_angle_rad)
        enemy_v0y = params.enemy_velocity * np.sin(enemy_angle_rad)

        # Posición inicial del misil enemigo
        enemy_x0 = params.defense_distance
        enemy_y0 = params.enemy_height

        # Calcular trayectoria del misil enemigo con efectos ambientales
        enemy_x = enemy_x0 + enemy_v0x * t
        enemy_y = enemy_y0 + enemy_v0y * t - 0.5 * params.constants.GRAVITY * np.power(t, 2)

        # Aplicar efectos del viento según las condiciones climáticas
        if params.weather.condition != WeatherCondition.CLEAR:
            wind_effect_x = params.weather.wind_speed * np.cos(np.radians(params.weather.wind_direction)) * t
            wind_effect_y = params.weather.wind_speed * np.sin(np.radians(params.weather.wind_direction)) * t

            # Añadir turbulencia si es necesario
            if params.weather.condition in [WeatherCondition.WINDY, WeatherCondition.STORMY]:
                turbulence = params.weather.turbulence_factor * params.weather.wind_speed
                random_turbulence_x = np.random.uniform(-turbulence, turbulence, size=len(t))
                random_turbulence_y = np.random.uniform(-turbulence, turbulence, size=len(t))
                wind_effect_x += random_turbulence_x
                wind_effect_y += random_turbulence_y

            enemy_x += wind_effect_x
            enemy_y += wind_effect_y

        # Asegurarse de que y no baje del suelo
        enemy_y = np.maximum(enemy_y, 0)

        # Trayectoria del antimisil (parabólica)
        # Velocidades iniciales
        v0x = params.antimissile_velocity * np.cos(antimissile_angle_rad)
        v0y = params.antimissile_velocity * np.sin(antimissile_angle_rad)

        # Vectorización avanzada
        time_mask = t >= params.antimissile_time_start
        delta_t = np.zeros_like(t)
        delta_t[time_mask] = t[time_mask] - params.antimissile_time_start

        # Trayectoria básica
        antimissile_x = v0x * delta_t
        antimissile_y = v0y * delta_t - 0.5 * params.constants.GRAVITY * np.power(delta_t, 2)

        # Aplicar efectos del viento según las condiciones climáticas
        if params.weather.condition != WeatherCondition.CLEAR and np.any(time_mask):
            active_delta_t = delta_t[time_mask]
            wind_effect_x = params.weather.wind_speed * np.cos(np.radians(params.weather.wind_direction)) * active_delta_t
            wind_effect_y = params.weather.wind_speed * np.sin(np.radians(params.weather.wind_direction)) * active_delta_t

            # Añadir turbulencia si es necesario
            if params.weather.condition in [WeatherCondition.WINDY, WeatherCondition.STORMY]:
                turbulence = params.weather.turbulence_factor * params.weather.wind_speed
                random_turbulence_x = np.random.uniform(-turbulence, turbulence, size=len(active_delta_t))
                random_turbulence_y = np.random.uniform(-turbulence, turbulence, size=len(active_delta_t))
                wind_effect_x += random_turbulence_x
                wind_effect_y += random_turbulence_y

            antimissile_x[time_mask] += wind_effect_x
            antimissile_y[time_mask] += wind_effect_y

        # Asegurarse de que y no baje del suelo
        antimissile_y = np.maximum(antimissile_y, 0)

        return t, enemy_x, enemy_y, antimissile_x, antimissile_y

    @staticmethod
    def calculate_trajectories_3d(params: SimulationParameters) -> Trajectory3D:
        """Calcula las trayectorias para ambos misiles en 3D"""
        # Tiempo para toda la simulación
        max_time = params.free_fall_time * 1.5
        t = np.arange(0, max_time, params.constants.TIME_STEP)

        # Convertir ángulos a radianes
        enemy_angle_rad = np.radians(params.enemy_angle)
        antimissile_angle_rad = np.radians(params.antimissile_angle)

        # Para 3D necesitamos un ángulo adicional (azimut)
        enemy_azimuth_rad = np.radians(random.uniform(0, 360))
        antimissile_azimuth_rad = np.radians(random.uniform(0, 360))

        # Trayectoria del misil enemigo
        enemy_v0x = params.enemy_velocity * np.cos(enemy_angle_rad) * np.cos(enemy_azimuth_rad)
        enemy_v0y = params.enemy_velocity * np.cos(enemy_angle_rad) * np.sin(enemy_azimuth_rad)
        enemy_v0z = params.enemy_velocity * np.sin(enemy_angle_rad)

        # Posición inicial del misil enemigo
        enemy_x0 = params.defense_distance
        enemy_y0 = 0  # En 3D, añadimos una dimensión
        enemy_z0 = params.enemy_height

        # Calcular trayectoria del misil enemigo
        enemy_x = enemy_x0 + enemy_v0x * t
        enemy_y = enemy_y0 + enemy_v0y * t
        enemy_z = enemy_z0 + enemy_v0z * t - 0.5 * params.constants.GRAVITY * np.power(t, 2)

        # Asegurarse de que z no baje del suelo
        enemy_z = np.maximum(enemy_z, 0)

        # Trayectoria del antimisil
        # Velocidades iniciales
        antimissile_v0x = params.antimissile_velocity * np.cos(antimissile_angle_rad) * np.cos(antimissile_azimuth_rad)
        antimissile_v0y = params.antimissile_velocity * np.cos(antimissile_angle_rad) * np.sin(antimissile_azimuth_rad)
        antimissile_v0z = params.antimissile_velocity * np.sin(antimissile_angle_rad)

        # Vectorización para el tiempo
        time_mask = t >= params.antimissile_time_start
        delta_t = np.zeros_like(t)
        delta_t[time_mask] = t[time_mask] - params.antimissile_time_start

        # Trayectoria básica
        antimissile_x = antimissile_v0x * delta_t
        antimissile_y = antimissile_v0y * delta_t
        antimissile_z = antimissile_v0z * delta_t - 0.5 * params.constants.GRAVITY * np.power(delta_t, 2)

        # Asegurarse de que z no baje del suelo
        antimissile_z = np.maximum(antimissile_z, 0)

        return t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z

    @staticmethod
    def check_interception(params: SimulationParameters,
                           t: np.ndarray,
                           enemy_x: np.ndarray,
                           enemy_y: np.ndarray,
                           antimissile_x: np.ndarray,
                           antimissile_y: np.ndarray) -> InterceptionResult:
        """Verifica si ocurre una intercepción y devuelve los detalles"""
        # Calcular la distancia entre misiles en cada paso de tiempo - optimización con numpy
        distances = np.sqrt(np.power(enemy_x - antimissile_x, 2) + np.power(enemy_y - antimissile_y, 2))

        # Encontrar la distancia mínima y el tiempo correspondiente
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        interception_time = t[min_distance_index]

        # Verificar si la intercepción ocurre dentro del margen de error
        interception = min_distance <= params.constants.ERROR_MARGIN

        # Verificar si la intercepción ocurre antes de que el misil enemigo golpee el suelo
        enemy_ground_indices = np.where(enemy_y <= 0)[0]

        if len(enemy_ground_indices) > 0:
            enemy_ground_time = t[enemy_ground_indices[0]]
            before_impact = interception_time < enemy_ground_time
        else:
            before_impact = True  # El enemigo nunca golpea el suelo en el marco de tiempo de la simulación

        # Obtener coordenadas de intercepción
        if interception and before_impact:
            intercept_x = enemy_x[min_distance_index]
            intercept_y = enemy_y[min_distance_index]
        else:
            intercept_x = None
            intercept_y = None

        return interception and before_impact, interception_time, min_distance, intercept_x, intercept_y

    @staticmethod
    def check_interception_3d(params: SimulationParameters,
                              t: np.ndarray,
                              enemy_x: np.ndarray,
                              enemy_y: np.ndarray,
                              enemy_z: np.ndarray,
                              antimissile_x: np.ndarray,
                              antimissile_y: np.ndarray,
                              antimissile_z: np.ndarray) -> InterceptionResult3D:
        """Verifica si ocurre una intercepción en 3D y devuelve los detalles"""
        # Calcular la distancia entre misiles en cada paso de tiempo en 3D
        distances = np.sqrt(np.power(enemy_x - antimissile_x, 2) +
                           np.power(enemy_y - antimissile_y, 2) +
                           np.power(enemy_z - antimissile_z, 2))

        # Encontrar la distancia mínima y el tiempo correspondiente
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        interception_time = t[min_distance_index]

        # Verificar si la intercepción ocurre dentro del margen de error
        interception = min_distance <= params.constants.ERROR_MARGIN

        # Verificar si la intercepción ocurre antes de que el misil enemigo golpee el suelo
        enemy_ground_indices = np.where(enemy_z <= 0)[0]

        if len(enemy_ground_indices) > 0:
            enemy_ground_time = t[enemy_ground_indices[0]]
            before_impact = interception_time < enemy_ground_time
        else:
            before_impact = True  # El enemigo nunca golpea el suelo en el marco de tiempo de la simulación

        # Obtener coordenadas de intercepción
        if interception and before_impact:
            intercept_x = enemy_x[min_distance_index]
            intercept_y = enemy_y[min_distance_index]
            intercept_z = enemy_z[min_distance_index]
        else:
            intercept_x = None
            intercept_y = None
            intercept_z = None

        return interception and before_impact, interception_time, min_distance, intercept_x, intercept_y, intercept_z

    @staticmethod
    def calculate_optimal_parameters(params: SimulationParameters) -> Tuple[float, float]:
        """Calcula los parámetros óptimos para la intercepción"""
        # Algoritmo optimizado para intercepción basada en física de misiles

        # Tiempo que tarda el misil enemigo en llegar al suelo
        enemy_angle_rad = np.radians(params.enemy_angle)
        enemy_v0y = params.enemy_velocity * np.sin(enemy_angle_rad)
        enemy_v0x = params.enemy_velocity * np.cos(enemy_angle_rad)

        # Resolver ecuación cuadrática para el tiempo de impacto del enemigo
        # y = h + v0y*t - 0.5*g*t^2 = 0
        a = 0.5 * params.constants.GRAVITY
        b = -enemy_v0y
        c = -params.enemy_height

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            # No hay solución real (no debería ocurrir en este contexto)
            return 45.0, 0.0

        # Tomamos la solución positiva
        enemy_time_to_ground = (-b + math.sqrt(discriminant)) / (2*a)

        # Calcular la posición horizontal donde el misil enemigo impactará
        enemy_impact_x = params.defense_distance + enemy_v0x * enemy_time_to_ground

        # Distancia horizontal que debe recorrer el antimisil
        distance_to_intercept = enemy_impact_x / 2  # Punto medio aproximado

        # Estimación del tiempo para llegar al punto de intercepción
        time_to_intercept = distance_to_intercept / (params.antimissile_velocity * 0.7)  # Factor de ajuste

        # Calcular la posición del misil enemigo en ese tiempo
        enemy_pos_x = params.defense_distance + enemy_v0x * time_to_intercept
        enemy_pos_y = params.enemy_height + enemy_v0y * time_to_intercept - 0.5 * params.constants.GRAVITY * time_to_intercept**2

        # Calcular el ángulo óptimo para interceptar
        delta_x = enemy_pos_x
        delta_y = enemy_pos_y

        # Ecuaciones para proyectil: encuentro tiempo de vuelo y ángulo necesario
        # Usamos el método iterativo para mayor precisión
        optimal_angle = 45.0  # Ángulo inicial
        optimal_time = 0.0

        for _ in range(5):  # Varias iteraciones para mejorar precisión
            # Calculo del ángulo basado en balística
            angle_rad = math.atan2(delta_y, delta_x)
            # Ajuste del ángulo para compensar la gravedad
            angle_rad += 0.1  # Pequeño ajuste empírico

            # Convertir a grados y limitar entre 0 y 90
            optimal_angle = math.degrees(angle_rad)
            optimal_angle = max(0, min(90, optimal_angle))

            # Estimar el tiempo óptimo de lanzamiento
            v0x = params.antimissile_velocity * math.cos(math.radians(optimal_angle))
            flight_time = delta_x / v0x if v0x > 0 else 0

            # El tiempo óptimo de lanzamiento es justo antes de la intercepción
            optimal_time = max(0, time_to_intercept - flight_time)

            # Recalcular posición del enemigo con el nuevo tiempo
            new_time_to_intercept = optimal_time + flight_time
            enemy_pos_x = params.defense_distance + enemy_v0x * new_time_to_intercept
            enemy_pos_y = params.enemy_height + enemy_v0y * new_time_to_intercept - 0.5 * params.constants.GRAVITY * new_time_to_intercept**2

            delta_x = enemy_pos_x
            delta_y = enemy_pos_y

        # Aplicar ajustes finales basados en condiciones climáticas
        if params.weather.condition != WeatherCondition.CLEAR:
            wind_effect = params.weather.wind_speed / 100.0
            wind_angle = math.radians(params.weather.wind_direction)

            # Ajustar ángulo según dirección del viento
            wind_compensation = wind_effect * math.sin(wind_angle - math.radians(optimal_angle))
            optimal_angle += wind_compensation * 5  # Factor de ajuste

            # Ajustar tiempo según intensidad del viento
            time_adjustment = wind_effect * 0.2
            optimal_time = max(0, optimal_time - time_adjustment)

        return optimal_angle, optimal_time

class SimulationEngine:
    """Motor de simulación para el sistema de defensa antimisiles"""

    def __init__(self):
        self.params = SimulationParameters()
        self.results = None
        self.state = SimulationState.INITIAL
        self._running = False
        self._last_simulation_time = 0
        self._cached_trajectories = None

    def initialize(self, params: SimulationParameters):
        """Inicializa el motor con los parámetros dados"""
        self.params = params
        self.state = SimulationState.INITIAL
        self._cached_trajectories = None

    def run_simulation(self) -> SimulationResults:
        """Ejecuta la simulación completa y devuelve los resultados"""
        self.state = SimulationState.RUNNING
        self._running = True
        start_time = time.time()

        try:
            # Calcular trayectorias según dimensiones
            if self.params.view_mode == ViewMode.MODE_2D:
                t, enemy_x, enemy_y, antimissile_x, antimissile_y = TrajectoryCalculator.calculate_trajectories(self.params)
                success, intercept_time, min_distance, intercept_x, intercept_y = TrajectoryCalculator.check_interception(
                    self.params, t, enemy_x, enemy_y, antimissile_x, antimissile_y
                )
                intercept_coords = (intercept_x, intercept_y, None) if intercept_x is not None else None

                # Calcular el punto de impacto del enemigo si no hay intercepción
                enemy_ground_indices = np.where(enemy_y <= 0)[0]
                if len(enemy_ground_indices) > 0:
                    enemy_impact_time = t[enemy_ground_indices[0]]
                    enemy_impact_x = enemy_x[enemy_ground_indices[0]]
                    enemy_impact_coords = (enemy_impact_x, 0, None)
                else:
                    enemy_impact_time = None
                    enemy_impact_coords = None

                # Calcular distancias recorridas
                if success:
                    time_index = np.where(t >= intercept_time)[0][0]
                    defense_missile_path = np.sum(np.sqrt(
                        np.diff(antimissile_x[:time_index+1])**2 +
                        np.diff(antimissile_y[:time_index+1])**2
                    ))
                    enemy_missile_path = np.sum(np.sqrt(
                        np.diff(enemy_x[:time_index+1])**2 +
                        np.diff(enemy_y[:time_index+1])**2
                    ))
                else:
                    defense_missile_path = np.sum(np.sqrt(
                        np.diff(antimissile_x)**2 +
                        np.diff(antimissile_y)**2
                    ))
                    enemy_missile_path = np.sum(np.sqrt(
                        np.diff(enemy_x)**2 +
                        np.diff(enemy_y)**2
                    ))

                # Cache para la animación
                self._cached_trajectories = (t, enemy_x, enemy_y, antimissile_x, antimissile_y)

            else:  # Modo 3D
                t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z = TrajectoryCalculator.calculate_trajectories_3d(self.params)
                success, intercept_time, min_distance, intercept_x, intercept_y, intercept_z = TrajectoryCalculator.check_interception_3d(
                    self.params, t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z
                )
                intercept_coords = (intercept_x, intercept_y, intercept_z) if intercept_x is not None else None

                # Calcular el punto de impacto del enemigo si no hay intercepción
                enemy_ground_indices = np.where(enemy_z <= 0)[0]
                if len(enemy_ground_indices) > 0:
                    enemy_impact_time = t[enemy_ground_indices[0]]
                    enemy_impact_x = enemy_x[enemy_ground_indices[0]]
                    enemy_impact_y = enemy_y[enemy_ground_indices[0]]
                    enemy_impact_coords = (enemy_impact_x, enemy_impact_y, 0)
                else:
                    enemy_impact_time = None
                    enemy_impact_coords = None

                # Calcular distancias recorridas
                if success:
                    time_index = np.where(t >= intercept_time)[0][0]
                    defense_missile_path = np.sum(np.sqrt(
                        np.diff(antimissile_x[:time_index+1])**2 +
                        np.diff(antimissile_y[:time_index+1])**2 +
                        np.diff(antimissile_z[:time_index+1])**2
                    ))
                    enemy_missile_path = np.sum(np.sqrt(
                        np.diff(enemy_x[:time_index+1])**2 +
                        np.diff(enemy_y[:time_index+1])**2 +
                        np.diff(enemy_z[:time_index+1])**2
                    ))
                else:
                    defense_missile_path = np.sum(np.sqrt(
                        np.diff(antimissile_x)**2 +
                        np.diff(antimissile_y)**2 +
                        np.diff(antimissile_z)**2
                    ))
                    enemy_missile_path = np.sum(np.sqrt(
                        np.diff(enemy_x)**2 +
                        np.diff(enemy_y)**2 +
                        np.diff(enemy_z)**2
                    ))

                # Cache para la animación
                self._cached_trajectories = (t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z)

            # Calcular resultados adicionales
            defense_response_time = self.params.antimissile_time_start
            fuel_consumption_rate = self.params.defense_missile.fuel_consumption
            fuel_consumed = fuel_consumption_rate * (intercept_time - self.params.antimissile_time_start) if success else 0

            # Calcular puntuación de eficiencia
            efficiency_base = 100 if success else 0
            time_penalty = max(0, intercept_time * 2)
            fuel_penalty = max(0, fuel_consumed / 10)
            efficiency_score = max(0, efficiency_base - time_penalty - fuel_penalty)

            # Calcular probabilidad de éxito basada en distancia mínima
            success_probability = 1.0 if min_distance <= self.params.constants.ERROR_MARGIN/2 else \
                                 max(0, 1 - (min_distance - self.params.constants.ERROR_MARGIN/2) / self.params.constants.ERROR_MARGIN)

            # Estimación de daños prevenidos
            casualties_prevented = None
            damage_prevented = None
            if success:
                explosive_yield = self.params.enemy_missile.explosive_yield
                casualties_prevented = int(explosive_yield * 0.5)  # Estimación simplificada
                damage_prevented = explosive_yield * 1000  # En miles de dólares

            # Crear objeto de resultados
            self.results = SimulationResults(
                success=success,
                intercept_time=float(intercept_time),
                min_distance=float(min_distance),
                intercept_coords=intercept_coords,
                enemy_impact_coords=enemy_impact_coords,
                enemy_time_to_impact=float(enemy_impact_time) if enemy_impact_time is not None else None,
                defense_response_time=float(defense_response_time),
                defense_missile_path_length=float(defense_missile_path),
                enemy_missile_path_length=float(enemy_missile_path),
                fuel_consumed=float(fuel_consumed),
                efficiency_score=float(efficiency_score),
                success_probability=float(success_probability),
                casualties_prevented=casualties_prevented,
                damage_prevented=damage_prevented
            )

            self.state = SimulationState.COMPLETED
            return self.results

        except Exception as e:
            self.state = SimulationState.FAILED
            print(f"Error en la simulación: {str(e)}")
            raise
        finally:
            self._running = False
            self._last_simulation_time = time.time() - start_time

    def get_cached_trajectories(self):
        """Devuelve las trayectorias calculadas en la última ejecución"""
        return self._cached_trajectories

    def optimize_parameters(self) -> Tuple[float, float]:
        """Calcula los parámetros óptimos para la intercepción"""
        return TrajectoryCalculator.calculate_optimal_parameters(self.params)

    def get_execution_time(self) -> float:
        """Devuelve el tiempo de ejecución de la última simulación"""
        return self._last_simulation_time

    def is_running(self) -> bool:
        """Indica si la simulación está en ejecución"""
        return self._running

    def cancel(self):
        """Cancela la simulación en curso"""
        self._running = False
        self.state = SimulationState.INITIAL

class SimulationVisualizer:
    """Clase para visualizar las simulaciones"""

    def __init__(self, master):
        self.master = master
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = None
        self.animation = None
        self.view_mode = ViewMode.MODE_2D
        self.theme = Theme.DARK
        self.show_grid = True
        self.show_trajectory = True
        self.show_velocity_vectors = False
        self.show_distance_markers = False
        self.show_impact_zone = True
        self.show_stats = True
        self.animation_speed = 1.0
        self.markers = []
        self.annotations = []
        self.colors = {
            "background": "#1E1E1E",
            "foreground": "#FFFFFF",
            "grid": "#333333",
            "enemy": "#FF0000",
            "defense": "#0000FF",
            "intercept": "#00FF00",
            "impact": "#FF6600",
            "text": "#CCCCCC"
        }

    def setup_canvas(self, parent_frame):
        """Configura el canvas para la visualización"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configurar eventos de mouse para interacción
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def apply_theme(self, theme: Theme):
        """Aplica un tema a la visualización"""
        self.theme = theme

        if theme == Theme.DARK:
            self.colors = {
                "background": "#1E1E1E",
                "foreground": "#FFFFFF",
                "grid": "#333333",
                "enemy": "#FF0000",
                "defense": "#0000FF",
                "intercept": "#00FF00",
                "impact": "#FF6600",
                "text": "#CCCCCC"
            }
        elif theme == Theme.LIGHT:
            self.colors = {
                "background": "#F5F5F5",
                "foreground": "#000000",
                "grid": "#CCCCCC",
                "enemy": "#CC0000",
                "defense": "#0000CC",
                "intercept": "#00CC00",
                "impact": "#FF6600",
                "text": "#333333"
            }
        elif theme == Theme.MILITARY:
            self.colors = {
                "background": "#272B2A",
                "foreground": "#A5F8D3",
                "grid": "#435C5D",
                "enemy": "#FF5733",
                "defense": "#33A1FF",
                "intercept": "#A2FF33",
                "impact": "#FFB733",
                "text": "#D6D6D6"
            }
        elif theme == Theme.SPACE:
            self.colors = {
                "background": "#0C0C2C",
                "foreground": "#E0E0FF",
                "grid": "#1E1E4C",
                "enemy": "#FF3333",
                "defense": "#3399FF",
                "intercept": "#33FF33",
                "impact": "#FFCC33",
                "text": "#BBBBFF"
            }
        elif theme == Theme.NEON:
            self.colors = {
                "background": "#121212",
                "foreground": "#FFFFFF",
                "grid": "#333333",
                "enemy": "#FF00FF",
                "defense": "#00FFFF",
                "intercept": "#00FF00",
                "impact": "#FFFF00",
                "text": "#CCCCCC"
            }

        # Aplicar tema a la figura actual
        self.figure.patch.set_facecolor(self.colors["background"])
        for ax in self.figure.get_axes():
            ax.set_facecolor(self.colors["background"])
            ax.spines['bottom'].set_color(self.colors["foreground"])
            ax.spines['top'].set_color(self.colors["foreground"])
            ax.spines['left'].set_color(self.colors["foreground"])
            ax.spines['right'].set_color(self.colors["foreground"])
            ax.tick_params(colors=self.colors["foreground"])
            ax.yaxis.label.set_color(self.colors["foreground"])
            ax.xaxis.label.set_color(self.colors["foreground"])
            ax.title.set_color(self.colors["foreground"])

            # Actualizar etiquetas y leyendas
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color(self.colors["foreground"])

            if ax.get_legend():
                for text in ax.get_legend().get_texts():
                    text.set_color(self.colors["foreground"])

        self.canvas.draw()

    def plot_simulation_2d(self, t, enemy_x, enemy_y, antimissile_x, antimissile_y, results=None):
        """Visualiza la simulación en 2D"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Configurar tema y estilo
        ax.set_facecolor(self.colors["background"])
        ax.grid(self.show_grid, color=self.colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)

        # Límites del gráfico con margen
        max_x = max(np.max(enemy_x), np.max(antimissile_x)) * 1.1
        max_y = max(np.max(enemy_y), np.max(antimissile_y)) * 1.1
        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])

        # Dibujar suelo
        ax.axhline(y=0, color=self.colors["foreground"], linestyle='-', linewidth=1)

        # Dibujar trayectorias si está habilitado
        if self.show_trajectory:
            # Trayectoria del misil enemigo
            ax.plot(enemy_x, enemy_y, '-', color=self.colors["enemy"], linewidth=1.5, alpha=0.7, label='Misil Enemigo')

            # Trayectoria del antimisil
            ax.plot(antimissile_x, antimissile_y, '-', color=self.colors["defense"], linewidth=1.5, alpha=0.7, label='Antimisil')

        # Mostrar posiciones iniciales
        ax.plot(enemy_x[0], enemy_y[0], 'o', color=self.colors["enemy"], markersize=8)
        ax.plot(antimissile_x[0], antimissile_y[0], 'o', color=self.colors["defense"], markersize=8)

        # Mostrar punto de intercepción si existe
        if results and results.success:
            if results.intercept_coords and results.intercept_coords[0] is not None:
                ax.plot(results.intercept_coords[0], results.intercept_coords[1], 'o',
                        color=self.colors["intercept"], markersize=10, label='Intercepción')

                # Dibujar círculo de explosión
                intercept_circle = plt.Circle((results.intercept_coords[0], results.intercept_coords[1]),
                                          500, color=self.colors["intercept"], fill=False, alpha=0.8, linestyle='--')
                ax.add_patch(intercept_circle)

        # Mostrar punto de impacto en caso de fallo
        elif results and not results.success and results.enemy_impact_coords:
            ax.plot(results.enemy_impact_coords[0], results.enemy_impact_coords[1], 'x',
                    color=self.colors["impact"], markersize=10, label='Impacto')

            # Dibujar zona de impacto
            if self.show_impact_zone:
                impact_circle = plt.Circle((results.enemy_impact_coords[0], 0),
                                          1000, color=self.colors["impact"], fill=True, alpha=0.3)
                ax.add_patch(impact_circle)

        # Mostrar estadísticas si está habilitado
        if self.show_stats and results:
            stats_text = f"Éxito: {'Sí' if results.success else 'No'}\n"
            stats_text += f"Tiempo: {results.intercept_time:.2f} s\n"
            stats_text += f"Distancia: {results.min_distance:.2f} m\n"
            stats_text += f"Eficiencia: {results.efficiency_score:.1f}%"

            ax.text(0.02, 0.96, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=self.colors["background"],
                              alpha=0.7, edgecolor=self.colors["foreground"]),
                    color=self.colors["text"])

        # Configurar etiquetas
        ax.set_xlabel('Distancia (metros)', color=self.colors["foreground"])
        ax.set_ylabel('Altura (metros)', color=self.colors["foreground"])
        ax.set_title('Simulación de Intercepción de Misiles', color=self.colors["foreground"])

        # Configurar leyenda
        ax.legend(loc='upper right', facecolor=self.colors["background"], edgecolor=self.colors["foreground"],
                  labelcolor=self.colors["foreground"])

        # Actualizar ticks
        ax.tick_params(axis='x', colors=self.colors["foreground"])
        ax.tick_params(axis='y', colors=self.colors["foreground"])

        self.canvas.draw()

    def plot_simulation_3d(self, t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z, results=None):
        """Visualiza la simulación en 3D"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        # Configurar tema y estilo
        ax.set_facecolor(self.colors["background"])
        ax.xaxis.pane.set_edgecolor(self.colors["foreground"])
        ax.yaxis.pane.set_edgecolor(self.colors["foreground"])
        ax.zaxis.pane.set_edgecolor(self.colors["foreground"])
        ax.xaxis.pane.set_alpha(0.7)
        ax.yaxis.pane.set_alpha(0.7)
        ax.zaxis.pane.set_alpha(0.7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(self.show_grid, color=self.colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)

        # Límites del gráfico con margen
        max_x = max(np.max(enemy_x), np.max(antimissile_x)) * 1.1
        max_y = max(np.max(enemy_y), np.max(antimissile_y)) * 1.1
        max_z = max(np.max(enemy_z), np.max(antimissile_z)) * 1.1
        ax.set_xlim([0, max_x])
        ax.set_ylim([-max_y/2, max_y/2])
        ax.set_zlim([0, max_z])

        # Dibujar suelo como plano
        x_grid, y_grid = np.meshgrid(np.linspace(0, max_x, 10), np.linspace(-max_y/2, max_y/2, 10))
        z_grid = np.zeros_like(x_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color=self.colors["foreground"], alpha=0.2)

        # Dibujar trayectorias si está habilitado
        if self.show_trajectory:
            # Trayectoria del misil enemigo
            ax.plot(enemy_x, enemy_y, enemy_z, '-', color=self.colors["enemy"], linewidth=2, alpha=0.7, label='Misil Enemigo')

            # Trayectoria del antimisil
            ax.plot(antimissile_x, antimissile_y, antimissile_z, '-', color=self.colors["defense"], linewidth=2, alpha=0.7, label='Antimisil')

        # Mostrar posiciones iniciales
        ax.plot([enemy_x[0]], [enemy_y[0]], [enemy_z[0]], 'o', color=self.colors["enemy"], markersize=8)
        ax.plot([antimissile_x[0]], [antimissile_y[0]], [antimissile_z[0]], 'o', color=self.colors["defense"], markersize=8)

        # Mostrar punto de intercepción si existe
        if results and results.success:
            if results.intercept_coords and results.intercept_coords[0] is not None:
                ax.plot([results.intercept_coords[0]], [results.intercept_coords[1]], [results.intercept_coords[2]], 'o',
                        color=self.colors["intercept"], markersize=10, label='Intercepción')

                # Representar esfera de explosión (usando puntos en una esfera)
                r = 300  # radio de la esfera
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = r * np.cos(u) * np.sin(v) + results.intercept_coords[0]
                y = r * np.sin(u) * np.sin(v) + results.intercept_coords[1]
                z = r * np.cos(v) + results.intercept_coords[2]
                ax.plot_wireframe(x, y, z, color=self.colors["intercept"], alpha=0.3)

        # Mostrar punto de impacto en caso de fallo
        elif results and not results.success and results.enemy_impact_coords:
            ax.plot([results.enemy_impact_coords[0]], [results.enemy_impact_coords[1]], [0], 'x',
                    color=self.colors["impact"], markersize=10, label='Impacto')

            # Dibujar zona de impacto
            if self.show_impact_zone:
                u, v = np.mgrid[0:2*np.pi:30j, 0:1:10j]
                radius = 1000 * v
                x = results.enemy_impact_coords[0] + radius * np.cos(u)
                y = results.enemy_impact_coords[1] + radius * np.sin(u)
                z = np.zeros_like(x)
                ax.plot_surface(x, y, z, color=self.colors["impact"], alpha=0.3)

        # Mostrar estadísticas si está habilitado
        if self.show_stats and results:
            stats_text = f"Éxito: {'Sí' if results.success else 'No'}\n"
            stats_text += f"Tiempo: {results.intercept_time:.2f} s\n"
            stats_text += f"Distancia: {results.min_distance:.2f} m\n"
            stats_text += f"Eficiencia: {results.efficiency_score:.1f}%"

            ax.text2D(0.02, 0.96, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=self.colors["background"],
                              alpha=0.7, edgecolor=self.colors["foreground"]),
                    color=self.colors["text"])

        # Configurar etiquetas
        ax.set_xlabel('Distancia (metros)', color=self.colors["foreground"])
        ax.set_ylabel('Lateral (metros)', color=self.colors["foreground"])
        ax.set_zlabel('Altura (metros)', color=self.colors["foreground"])
        ax.set_title('Simulación de Intercepción de Misiles', color=self.colors["foreground"])

        # Configurar leyenda
        ax.legend(loc='upper right', facecolor=self.colors["background"], edgecolor=self.colors["foreground"],
                  labelcolor=self.colors["foreground"])

        # Actualizar ticks
        ax.tick_params(axis='x', colors=self.colors["foreground"])
        ax.tick_params(axis='y', colors=self.colors["foreground"])
        ax.tick_params(axis='z', colors=self.colors["foreground"])

        self.canvas.draw()

    def on_click(self, event):
        """Manejador de eventos para clics del mouse"""
        pass

    def on_scroll(self, event):
        """Manejador de eventos para el desplazamiento del mouse"""
        pass

    def on_hover(self, event):
        """Manejador de eventos para el movimiento del mouse"""
        pass

class ResultsFormatter:
    """Clase para formatear los resultados de la simulación"""

    @staticmethod
    def format_interception_results(
        interception: bool,
        intercept_time: float,
        min_distance: float,
        intercept_x: Optional[float],
        intercept_y: Optional[float],
        params: SimulationParameters
    ) -> Tuple[str, str]:
        """Formatea los resultados de la intercepción para mostrar"""
        if interception:
            result_text = (
                f"¡INTERCEPCIÓN EXITOSA!\n\n"
                f"Tiempo de intercepción: {intercept_time:.2f} s\n"
                f"Distancia mínima: {min_distance:.2f} m\n"
                f"Coordenadas: ({intercept_x/1000:.2f} km, {intercept_y/1000:.2f} km)\n\n"
                f"Parámetros utilizados:\n"
                f"• Ángulo: {params.antimissile_angle:.2f}°\n"
                f"• Velocidad: {params.antimissile_velocity:.2f} m/s\n"
                f"• Tiempo de lanzamiento: {params.antimissile_time_start:.2f} s"
            )
            status_text = f"¡Intercepción exitosa en t={intercept_time:.2f}s!"
        else:
            result_text = (
                f"NO SE LOGRÓ LA INTERCEPCIÓN\n\n"
                f"Distancia mínima: {min_distance:.2f} m\n"
                f"Se requiere ajustar los parámetros:\n"
                f"• Ángulo: {params.antimissile_angle:.2f}°\n"
                f"• Velocidad: {params.antimissile_velocity:.2f} m/s\n"
                f"• Tiempo de lanzamiento: {params.antimissile_time_start:.2f} s\n\n"
                f"Pruebe con 'Calcular Parámetros Óptimos'"
            )
            status_text = "No se logró la intercepción. Ajuste los parámetros."

        return result_text, status_text

class ThemeManager:
    """Gestor del tema de la aplicación"""

    def __init__(self, initial_theme: Theme = Theme.DARK):
        self.current_theme = initial_theme
        self._apply_theme()

    def toggle_theme(self) -> Theme:
        """Cambia entre tema claro y oscuro"""
        self.current_theme = Theme.LIGHT if self.current_theme == Theme.DARK else Theme.DARK
        self._apply_theme()
        return self.current_theme

    def _apply_theme(self) -> None:
        """Aplica el tema actual a la interfaz"""
        theme_name = "dark" if self.current_theme == Theme.DARK else "light"
        sv_ttk.set_theme(theme_name)

    @property
    def is_dark(self) -> bool:
        """Devuelve True si el tema actual es oscuro"""
        return self.current_theme == Theme.DARK

class PlotManager:
    """Clase para gestionar la visualización de gráficos"""

    def __init__(self, fig: Figure, ax: plt.Axes, canvas: FigureCanvasTkAgg, theme_manager: ThemeManager):
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.theme_manager = theme_manager
        self._update_theme()

    def _update_theme(self) -> None:
        """Actualiza el tema del gráfico"""
        is_dark = self.theme_manager.is_dark

        # Configurar colores según el tema
        bg_color = '#2e2e2e' if is_dark else 'white'
        plot_bg_color = '#3e3e3e' if is_dark else '#f0f0f0'

        self.fig.set_facecolor(bg_color)
        self.ax.set_facecolor(plot_bg_color)
        self.canvas.draw()

    @property
    def text_color(self) -> str:
        """Devuelve el color del texto según el tema"""
        return 'white' if self.theme_manager.is_dark else 'black'

    @property
    def grid_color(self) -> str:
        """Devuelve el color de la cuadrícula según el tema"""
        return '#555555' if self.theme_manager.is_dark else '#cccccc'

    @property
    def legend_color(self) -> str:
        """Devuelve el color de fondo de la leyenda según el tema"""
        return '#333333' if self.theme_manager.is_dark else '#f0f0f0'

    def on_theme_change(self) -> None:
        """Manejador para cambios de tema"""
        self._update_theme()

    def plot_initial_scenario(self, params: SimulationParameters) -> None:
        """Dibuja el escenario inicial sin trayectorias"""
        self.ax.clear()

        # Posición del misil enemigo (punto C)
        self.ax.plot(params.defense_distance, params.enemy_height, 'ro', markersize=10, label='Misil Enemigo (C)')

        # Posición del sistema de defensa (punto A)
        self.ax.plot(0, 0, 'bo', markersize=10, label='Sistema Antiaéreo (A)')

        # Posición del objetivo (punto B)
        self.ax.plot(params.defense_distance, 0, 'go', markersize=10, label='Objetivo (B)')

        # Dibujar línea discontinua que muestra la trayectoria
        self.ax.plot([params.defense_distance, params.defense_distance], [0, params.enemy_height], 'r--', alpha=0.5)

        self._configure_plot_style(params)
        self.canvas.draw()

    def plot_full_simulation(self, params: SimulationParameters,
                             t: np.ndarray, enemy_x: np.ndarray, enemy_y: np.ndarray,
                             antimissile_x: np.ndarray, antimissile_y: np.ndarray,
                             interception: bool = False, intercept_x: Optional[float] = None,
                             intercept_y: Optional[float] = None) -> None:
        """Dibuja la simulación completa con trayectorias"""
        self.ax.clear()

        # Dibujar trayectorias con estilos mejorados
        self.ax.plot(enemy_x, enemy_y, 'r-', label='Trayectoria Misil Enemigo', linewidth=2)
        self.ax.plot(antimissile_x, antimissile_y, 'b-', label='Trayectoria Misil Antiaéreo', linewidth=2)

        # Dibujar posiciones
        self.ax.plot(0, 0, 'bo', markersize=10, label='Sistema Antiaéreo (A)')
        self.ax.plot(params.defense_distance, 0, 'go', markersize=10, label='Objetivo (B)')
        self.ax.plot(params.defense_distance, params.enemy_height, 'ro', markersize=10, label='Posición Inicial Misil Enemigo (C)')

        # Marcar punto de intercepción si existe
        if interception and intercept_x is not None and intercept_y is not None:
            self.ax.plot(intercept_x, intercept_y, 'mo', markersize=12, label='Punto de Intercepción')
            self.ax.add_patch(plt.Circle((intercept_x, intercept_y), params.constants.ERROR_MARGIN, fill=False,
                                       color='m', linestyle='--', alpha=0.7))

        self._configure_plot_style(params, max_x=np.max(antimissile_x), max_y=np.max(antimissile_y))
        self.canvas.draw()

    def prepare_animation(self, params: SimulationParameters) -> Tuple[plt.Line2D, plt.Line2D]:
        """Prepara el gráfico para animación"""
        self.ax.clear()

        # Dibujar posiciones iniciales
        self.ax.plot(0, 0, 'bo', markersize=10, label='Sistema Antiaéreo (A)')
        self.ax.plot(params.defense_distance, 0, 'go', markersize=10, label='Objetivo (B)')
        self.ax.plot(params.defense_distance, params.enemy_height, 'ro', markersize=10, label='Posición Inicial Misil Enemigo (C)')

        self._configure_plot_style(params)

        # Crear líneas vacías para las trayectorias con estilos mejorados
        enemy_path, = self.ax.plot([], [], 'r-', linewidth=1.5, alpha=0.7)
        antimissile_path, = self.ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7)

        self.canvas.draw()
        return enemy_path, antimissile_path

    def _configure_plot_style(self, params: SimulationParameters, max_x: Optional[float] = None, max_y: Optional[float] = None) -> None:
        """Configura el estilo del gráfico"""
        # Configurar límites de los ejes con margen
        x_max = max(params.defense_distance * 1.1, max_x * 1.1 if max_x is not None else 0, 60000)
        y_max = max(params.enemy_height * 1.1, max_y * 1.1 if max_y is not None else 0, 15000)

        self.ax.set_xlim([-5000, x_max])
        self.ax.set_ylim([-500, y_max])

        # Etiquetas y cuadrícula con estilos personalizados
        self.ax.set_xlabel('Distancia Horizontal (m)', color=self.text_color, fontsize=10)
        self.ax.set_ylabel('Altura (m)', color=self.text_color, fontsize=10)
        self.ax.set_title('Simulación de Intercepción de Misiles', color=self.text_color, fontsize=14, fontweight='bold')
        self.ax.grid(True, color=self.grid_color, linestyle='--', alpha=0.7)

        # Actualizar colores de las marcas
        self.ax.tick_params(axis='x', colors=self.text_color)
        self.ax.tick_params(axis='y', colors=self.text_color)

        # Leyenda con fondo personalizado y estilo moderno
        legend = self.ax.legend(facecolor=self.legend_color, framealpha=0.9)
        for text in legend.get_texts():
            text.set_color(self.text_color)

        # Convertir ejes a km para mejor legibilidad
        def format_coord(x: float, y: float) -> str:
            return f'x={x/1000:.2f} km, y={y/1000:.2f} km'

        self.ax.format_coord = format_coord

class PresetManager:
    """Gestor de escenarios predefinidos"""

    # Constantes para los presets
    STANDARD_SCENARIO: Final[Dict[str, float]] = {"defense_distance": 30000, "enemy_height": 10000}
    CLOSE_DEFENSE: Final[Dict[str, float]] = {"defense_distance": 15000, "enemy_height": 10000}
    FAR_DEFENSE: Final[Dict[str, float]] = {"defense_distance": 50000, "enemy_height": 10000}
    HIGH_ALTITUDE: Final[Dict[str, float]] = {"defense_distance": 30000, "enemy_height": 15000}

    @classmethod
    def apply_preset(cls, params: SimulationParameters, preset_name: str) -> None:
        """Aplica un preset a los parámetros de simulación"""
        presets = {
            "standard": cls.STANDARD_SCENARIO,
            "close": cls.CLOSE_DEFENSE,
            "far": cls.FAR_DEFENSE,
            "high": cls.HIGH_ALTITUDE
        }

        if preset_name in presets:
            for key, value in presets[preset_name].items():
                setattr(params, key, value)

class UIManager:
    """Gestor para componentes de la interfaz de usuario"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.style = ttk.Style()
        self._configure_styles()

    def _configure_styles(self) -> None:
        """Configura estilos personalizados para widgets"""
        # Personalizar etiquetas
        self.style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 12))
        self.style.configure('Info.TLabel', font=('Segoe UI', 10))
        self.style.configure('Status.TLabel', font=('Segoe UI', 10, 'italic'))

        # Personalizar botones
        self.style.configure('Primary.TButton', font=('Segoe UI', 11))
        self.style.configure('Secondary.TButton', font=('Segoe UI', 10))
        self.style.configure('Danger.TButton', font=('Segoe UI', 10), foreground='red')

        # Personalizar frames
        self.style.configure('Card.TFrame', relief='raised')

    def create_tooltip(self, widget: tk.Widget, text: str) -> None:
        """Crea un tooltip para un widget"""

        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            # Crear ventana emergente
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")

            label = ttk.Label(self.tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief="solid", borderwidth=1,
                            font=("Segoe UI", 9, "normal"))
            label.pack(ipadx=3, ipady=2)

        def leave(event):
            if hasattr(self, "tooltip"):
                self.tooltip.destroy()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

class AnimationController:
    """Controlador para animaciones de la simulación"""

    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start_animation(self, animation_func: Callable[[], None]) -> None:
        """Inicia una animación en un hilo separado"""
        if self.running:
            return

        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run_animation, args=(animation_func,))
        self.thread.daemon = True  # El hilo se cerrará cuando el programa principal termine
        self.thread.start()

    def _run_animation(self, animation_func: Callable[[], None]) -> None:
        """Ejecuta la función de animación y maneja la finalización"""
        try:
            animation_func()
        finally:
            self.running = False

    def stop_animation(self) -> None:
        """Detiene la animación en curso"""
        if not self.running:
            return

        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    @property
    def is_running(self) -> bool:
        """Devuelve si la animación está en ejecución"""
        return self.running

class MissileInterceptionApp:
    """Aplicación principal para la simulación de intercepción de misiles"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simulación Avanzada de Intercepción de Misiles")
        self.root.geometry("1280x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Inicializar componentes principales
        self.params = SimulationParameters()
        self.theme_manager = ThemeManager()
        self.ui_manager = UIManager(root)
        self.animation_controller = AnimationController()
        self.simulation_state = SimulationState.INITIAL

        # Crear variables de control para la interfaz
        self._create_control_variables()

        # Crear la GUI
        self._create_widgets()

        # Inicializar gráfico
        self._setup_plot()

        # Inicializar variables de la GUI con valores predeterminados
        self._update_gui_from_params()

        # Plot inicial
        self.plot_manager.plot_initial_scenario(self.params)

        # Configurar temporizador para actualización periódica (para animaciones fluidas)
        self.root.after(100, self._periodic_update)

    def _create_control_variables(self) -> None:
        """Crea las variables de control para la interfaz"""
        self.enemy_height_var = tk.DoubleVar()
        self.defense_distance_var = tk.DoubleVar()
        self.antimissile_velocity_var = tk.DoubleVar()
        self.antimissile_angle_var = tk.DoubleVar()
        self.antimissile_time_var = tk.DoubleVar()
        self.animation_speed_var = tk.DoubleVar()
        self.results_var = tk.StringVar(value="Resultados de la simulación aparecerán aquí.")
        self.status_var = tk.StringVar(value="Sistema listo para simulación.")
        self.view_mode_var = tk.StringVar(value="2D")

    def _setup_plot(self) -> None:
        """Configura el área de gráficos"""
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Crear canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicializar el gestor de gráficos
        self.plot_manager = PlotManager(self.fig, self.ax, self.canvas, self.theme_manager)

    def _create_widgets(self) -> None:
        """Crea todos los widgets para la interfaz gráfica"""
        # Configuración básica
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Crear la barra superior
        self._create_top_bar()

        # Crear panel de control
        self._create_control_panel()

        # Crear panel de visualización
        self._create_visualization_panel()

        # Crear barra de estado
        self._create_status_bar()

    def _create_top_bar(self) -> None:
        """Crea la barra superior con título y controles globales"""
        title_frame = ttk.Frame(self.main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            title_frame,
            text="Simulación Avanzada de Intercepción de Misiles",
            style='Title.TLabel'
        ).pack(side=tk.LEFT)

        # Botón de cambio de tema
        theme_button = ttk.Button(
            title_frame,
            text="Cambiar Tema",
            command=self.toggle_theme
        )
        theme_button.pack(side=tk.RIGHT, padx=5)

    def _create_control_panel(self) -> None:
        """Crea el panel de control con todos los parámetros"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Panel de Control", padding="15")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=10)

        # Parámetros del misil enemigo
        self._create_enemy_controls(control_frame)

        # Parámetros del sistema de defensa
        self._create_defense_controls(control_frame)

        # Parámetros de misil antiaéreo
        self._create_antimissile_controls(control_frame)

        # Tiempo de lanzamiento
        self._create_timing_controls(control_frame)

        # Velocidad de animación
        self._create_animation_controls(control_frame)

        # Selector de modo de vista
        self._create_view_mode_selector(control_frame)

        # Botones de acción
        self._create_action_buttons(control_frame)

    def _create_enemy_controls(self, parent: ttk.Frame) -> None:
        """Crea los controles para los parámetros del misil enemigo"""
        enemy_frame = ttk.LabelFrame(parent, text="Parámetros del Misil Enemigo", style='Card.TFrame')
        enemy_frame.pack(fill=tk.X, pady=5)

        ttk.Label(enemy_frame, text="Altura del Misil Enemigo (m):", style='Subtitle.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(enemy_frame, textvariable=self.enemy_height_var).grid(row=0, column=1, padx=5, pady=2)

    def _create_defense_controls(self, parent: ttk.Frame) -> None:
        """Crea los controles para los parámetros del sistema de defensa"""
        defense_frame = ttk.LabelFrame(parent, text="Parámetros del Sistema de Defensa", style='Card.TFrame')
        defense_frame.pack(fill=tk.X, pady=5)

        ttk.Label(defense_frame, text="Distancia al Objetivo (m):", style='Subtitle.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(defense_frame, textvariable=self.defense_distance_var).grid(row=0, column=1, padx=5, pady=2)

    def _create_antimissile_controls(self, parent: ttk.Frame) -> None:
        """Crea los controles para los parámetros del misil antiaéreo"""
        antimissile_frame = ttk.LabelFrame(parent, text="Parámetros del Misil Antiaéreo", style='Card.TFrame')
        antimissile_frame.pack(fill=tk.X, pady=5)

        ttk.Label(antimissile_frame, text="Velocidad del Misil Antiaéreo (m/s):", style='Subtitle.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(antimissile_frame, textvariable=self.antimissile_velocity_var).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(antimissile_frame, text="Ángulo de Lanzamiento (grados):", style='Subtitle.TLabel').grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(antimissile_frame, textvariable=self.antimissile_angle_var).grid(row=1, column=1, padx=5, pady=2)

    def _create_timing_controls(self, parent: ttk.Frame) -> None:
        """Crea los controles para el tiempo de lanzamiento"""
        timing_frame = ttk.LabelFrame(parent, text="Tiempo de Lanzamiento", style='Card.TFrame')
        timing_frame.pack(fill=tk.X, pady=5)

        ttk.Label(timing_frame, text="Tiempo de Lanzamiento (s):", style='Subtitle.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(timing_frame, textvariable=self.antimissile_time_var).grid(row=0, column=1, padx=5, pady=2)

    def _create_animation_controls(self, parent: ttk.Frame) -> None:
        """Crea los controles para la velocidad de animación"""
        animation_frame = ttk.LabelFrame(parent, text="Velocidad de Animación", style='Card.TFrame')
        animation_frame.pack(fill=tk.X, pady=5)

        ttk.Label(animation_frame, text="Multiplicador de Velocidad:", style='Subtitle.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(animation_frame, textvariable=self.animation_speed_var).grid(row=0, column=1, padx=5, pady=2)

    def _create_view_mode_selector(self, parent: ttk.Frame) -> None:
        """Crea el selector de modo de vista"""
        view_mode_frame = ttk.LabelFrame(parent, text="Modo de Vista", style='Card.TFrame')
        view_mode_frame.pack(fill=tk.X, pady=5)

        ttk.Label(view_mode_frame, text="Seleccionar Modo de Vista:", style='Subtitle.TLabel').grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Combobox(view_mode_frame, textvariable=self.view_mode_var, values=["2D", "3D"], state="readonly").grid(row=0, column=1, padx=5, pady=2)

    def _create_action_buttons(self, parent: ttk.Frame) -> None:
        """Crea los botones de acción"""
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="Ejecutar Simulación", command=self.run_simulation, style='Primary.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Animar Simulación", command=self.animate_simulation, style='Primary.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Calcular Parámetros Óptimos", command=self.calculate_optimal_parameters, style='Secondary.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Reiniciar Simulación", command=self.reset_simulation, style='Danger.TButton').pack(fill=tk.X, pady=2)

    def _create_visualization_panel(self) -> None:
        """Crea el panel de visualización para el gráfico"""
        self.graph_frame = ttk.LabelFrame(self.main_frame, text="Visualización de la Simulación", padding="10")
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_status_bar(self) -> None:
        """Crea la barra de estado en la parte inferior"""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        ttk.Label(status_frame, textvariable=self.status_var, style='Status.TLabel').pack(side=tk.LEFT, padx=5)

        self.results_text = tk.Text(status_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def _update_gui_from_params(self) -> None:
        """Actualiza la interfaz gráfica con los valores de los parámetros"""
        self.enemy_height_var.set(self.params.enemy_height)
        self.defense_distance_var.set(self.params.defense_distance)
        self.antimissile_velocity_var.set(self.params.antimissile_velocity)
        self.antimissile_angle_var.set(self.params.antimissile_angle)
        self.antimissile_time_var.set(self.params.antimissile_time_start)
        self.animation_speed_var.set(self.params.animation_speed)
        self.view_mode_var.set("2D" if self.params.view_mode == ViewMode.MODE_2D else "3D")

    def _update_params_from_gui(self) -> None:
        """Actualiza los parámetros de simulación desde la interfaz gráfica"""
        self.params.enemy_height = self.enemy_height_var.get()
        self.params.defense_distance = self.defense_distance_var.get()
        self.params.antimissile_velocity = self.antimissile_velocity_var.get()
        self.params.antimissile_angle = self.antimissile_angle_var.get()
        self.params.antimissile_time_start = self.antimissile_time_var.get()
        self.params.animation_speed = self.animation_speed_var.get()
        self.params.view_mode = ViewMode.MODE_2D if self.view_mode_var.get() == "2D" else ViewMode.MODE_3D

    def toggle_theme(self) -> None:
        """Cambia el tema de la interfaz"""
        new_theme = self.theme_manager.toggle_theme()
        self.plot_manager.on_theme_change()
        self.status_var.set(f"Tema cambiado a {'oscuro' if new_theme == Theme.DARK else 'claro'}.")

    def run_simulation(self) -> None:
        """Ejecuta la simulación y muestra los resultados"""
        self._update_params_from_gui()
        if self.params.view_mode == ViewMode.MODE_2D:
            t, enemy_x, enemy_y, antimissile_x, antimissile_y = TrajectoryCalculator.calculate_trajectories(self.params)
            interception, intercept_time, min_distance, intercept_x, intercept_y = TrajectoryCalculator.check_interception(
                self.params, t, enemy_x, enemy_y, antimissile_x, antimissile_y
            )

            result_text, status_text = ResultsFormatter.format_interception_results(
                interception, intercept_time, min_distance, intercept_x, intercept_y, self.params
            )

            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            self.results_text.config(state=tk.DISABLED)
            self.status_var.set(status_text)

            self.plot_manager.plot_full_simulation(self.params, t, enemy_x, enemy_y, antimissile_x, antimissile_y, interception, intercept_x, intercept_y)
        else:
            t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z = TrajectoryCalculator.calculate_trajectories_3d(self.params)
            interception, intercept_time, min_distance, intercept_x, intercept_y, intercept_z = TrajectoryCalculator.check_interception_3d(
                self.params, t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z
            )

            result_text, status_text = ResultsFormatter.format_interception_results(
                interception, intercept_time, min_distance, intercept_x, intercept_y, self.params
            )

            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            self.results_text.config(state=tk.DISABLED)
            self.status_var.set(status_text)

            self.plot_manager.plot_full_simulation(self.params, t, enemy_x, enemy_y, antimissile_x, antimissile_y, interception, intercept_x, intercept_y)

        self.simulation_state = SimulationState.COMPLETED

    def animate_simulation(self) -> None:
        """Anima la simulación paso a paso"""
        self._update_params_from_gui()
        self.simulation_state = SimulationState.ANIMATING
        self.status_var.set("Animación en progreso...")

        if self.params.view_mode == ViewMode.MODE_2D:
            t, enemy_x, enemy_y, antimissile_x, antimissile_y = TrajectoryCalculator.calculate_trajectories(self.params)
            enemy_path, antimissile_path = self.plot_manager.prepare_animation(self.params)

            def animate():
                for i in range(len(t)):
                    if self.animation_controller.stop_event.is_set():
                        break
                    enemy_path.set_data(enemy_x[:i], enemy_y[:i])
                    antimissile_path.set_data(antimissile_x[:i], antimissile_y[:i])
                    self.canvas.draw()
                    time.sleep(self.params.constants.TIME_STEP / self.params.animation_speed)

                interception, intercept_time, min_distance, intercept_x, intercept_y = TrajectoryCalculator.check_interception(
                    self.params, t, enemy_x, enemy_y, antimissile_x, antimissile_y
                )

                result_text, status_text = ResultsFormatter.format_interception_results(
                    interception, intercept_time, min_distance, intercept_x, intercept_y, self.params
                )

                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, result_text)
                self.results_text.config(state=tk.DISABLED)
                self.status_var.set(status_text)

                self.plot_manager.plot_full_simulation(self.params, t, enemy_x, enemy_y, antimissile_x, antimissile_y, interception, intercept_x, intercept_y)
                self.simulation_state = SimulationState.COMPLETED

            self.animation_controller.start_animation(animate)
        else:
            t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z = TrajectoryCalculator.calculate_trajectories_3d(self.params)
            enemy_path, antimissile_path = self.plot_manager.prepare_animation(self.params)

            def animate():
                for i in range(len(t)):
                    if self.animation_controller.stop_event.is_set():
                        break
                    enemy_path.set_data(enemy_x[:i], enemy_y[:i])
                    antimissile_path.set_data(antimissile_x[:i], antimissile_y[:i])
                    self.canvas.draw()
                    time.sleep(self.params.constants.TIME_STEP / self.params.animation_speed)

                interception, intercept_time, min_distance, intercept_x, intercept_y, intercept_z = TrajectoryCalculator.check_interception_3d(
                    self.params, t, enemy_x, enemy_y, enemy_z, antimissile_x, antimissile_y, antimissile_z
                )

                result_text, status_text = ResultsFormatter.format_interception_results(
                    interception, intercept_time, min_distance, intercept_x, intercept_y, self.params
                )

                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, result_text)
                self.results_text.config(state=tk.DISABLED)
                self.status_var.set(status_text)

                self.plot_manager.plot_full_simulation(self.params, t, enemy_x, enemy_y, antimissile_x, antimissile_y, interception, intercept_x, intercept_y)
                self.simulation_state = SimulationState.COMPLETED

            self.animation_controller.start_animation(animate)

    def calculate_optimal_parameters(self) -> None:
        """Calcula y aplica los parámetros óptimos para la intercepción"""
        optimal_angle, optimal_time = TrajectoryCalculator.calculate_optimal_parameters(self.params)
        self.params.antimissile_angle = optimal_angle
        self.params.antimissile_time_start = optimal_time
        self._update_gui_from_params()
        self.status_var.set("Parámetros óptimos calculados y aplicados.")

    def reset_simulation(self) -> None:
        """Reinicia la simulación a los valores predeterminados"""
        self.params = SimulationParameters()
        self._update_gui_from_params()
        self.plot_manager.plot_initial_scenario(self.params)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.status_var.set("Simulación reiniciada.")
        self.simulation_state = SimulationState.INITIAL

    def on_closing(self) -> None:
        """Manejador para el cierre de la ventana"""
        if self.animation_controller.is_running:
            self.animation_controller.stop_animation()
        self.root.destroy()

    def _periodic_update(self) -> None:
        """Actualización periódica para manejar eventos y animaciones"""
        if self.simulation_state == SimulationState.ANIMATING and not self.animation_controller.is_running:
            self.simulation_state = SimulationState.COMPLETED
        self.root.after(100, self._periodic_update)

if __name__ == "__main__":
    root = tk.Tk()
    app = MissileInterceptionApp(root)
    root.mainloop()