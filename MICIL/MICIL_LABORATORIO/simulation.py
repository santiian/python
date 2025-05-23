import numpy as np
import math
import random
import time
from typing import Tuple, List, Optional, Dict, Any, Callable, TypeAlias, Final
from models import ParametrosSimulacion, ResultadosSimulacion, CondicionClima, ModoVista, EstadoSimulacion

class CalculadorTrayectorias:
    """Clase para calcular trayectorias de misiles"""

    @staticmethod
    def calcular_trayectorias(params: ParametrosSimulacion) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcula las trayectorias para ambos misiles en 2D."""
        tiempo_maximo = params.tiempo_caida_libre * 1.5
        t = np.arange(0, tiempo_maximo, params.constantes.PASO_TIEMPO)

        angulo_enemigo_rad = np.radians(params.angulo_enemigo)
        angulo_antimisil_rad = np.radians(params.angulo_antimisil)

        enemigo_v0x, enemigo_v0y = params.velocidad_enemigo * np.cos(angulo_enemigo_rad), params.velocidad_enemigo * np.sin(angulo_enemigo_rad)
        enemigo_x0, enemigo_y0 = params.distancia_defensa, params.altura_enemigo

        enemigo_x = enemigo_x0 + enemigo_v0x * t
        enemigo_y = enemigo_y0 + enemigo_v0y * t - 0.5 * params.constantes.GRAVEDAD * np.power(t, 2)

        if params.clima.condicion != CondicionClima.DESPEJADO:
            efecto_viento_x, efecto_viento_y = CalculadorTrayectorias._aplicar_efectos_viento(params, t)
            enemigo_x += efecto_viento_x
            enemigo_y += efecto_viento_y

        enemigo_y = np.maximum(enemigo_y, 0)

        v0x, v0y = params.velocidad_antimisil * np.cos(angulo_antimisil_rad), params.velocidad_antimisil * np.sin(angulo_antimisil_rad)
        mascara_tiempo = t >= params.tiempo_inicio_antimisil
        delta_t = np.zeros_like(t)
        delta_t[mascara_tiempo] = t[mascara_tiempo] - params.tiempo_inicio_antimisil

        antimisil_x, antimisil_y = v0x * delta_t, v0y * delta_t - 0.5 * params.constantes.GRAVEDAD * np.power(delta_t, 2)

        if params.clima.condicion != CondicionClima.DESPEJADO and np.any(mascara_tiempo):
            delta_t_activo = delta_t[mascara_tiempo]
            efecto_viento_x, efecto_viento_y = CalculadorTrayectorias._aplicar_efectos_viento(params, delta_t_activo)
            antimisil_x[mascara_tiempo] += efecto_viento_x
            antimisil_y[mascara_tiempo] += efecto_viento_y

        antimisil_y = np.maximum(antimisil_y, 0)

        return t, enemigo_x, enemigo_y, antimisil_x, antimisil_y

    @staticmethod
    def _aplicar_efectos_viento(params: ParametrosSimulacion, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica efectos del viento a las trayectorias."""
        efecto_viento_x = params.clima.velocidad_viento * np.cos(np.radians(params.clima.direccion_viento)) * t
        efecto_viento_y = params.clima.velocidad_viento * np.sin(np.radians(params.clima.direccion_viento)) * t

        if params.clima.condicion in [CondicionClima.VENTOSO, CondicionClima.TORMENTOSO]:
            turbulencia = params.clima.factor_turbulencia * params.clima.velocidad_viento
            turbulencia_aleatoria_x = np.random.uniform(-turbulencia, turbulencia, size=len(t))
            turbulencia_aleatoria_y = np.random.uniform(-turbulencia, turbulencia, size=len(t))
            efecto_viento_x += turbulencia_aleatoria_x
            efecto_viento_y += turbulencia_aleatoria_y

        return efecto_viento_x, efecto_viento_y

    @staticmethod
    def calcular_trayectorias_3d(params: ParametrosSimulacion) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcula las trayectorias para ambos misiles en 3D"""
        tiempo_maximo = params.tiempo_caida_libre * 1.5
        t = np.arange(0, tiempo_maximo, params.constantes.PASO_TIEMPO)

        angulo_enemigo_rad = np.radians(params.angulo_enemigo)
        angulo_antimisil_rad = np.radians(params.angulo_antimisil)

        azimut_enemigo_rad = np.radians(random.uniform(0, 360))
        azimut_antimisil_rad = np.radians(random.uniform(0, 360))

        enemigo_v0x = params.velocidad_enemigo * np.cos(angulo_enemigo_rad) * np.cos(azimut_enemigo_rad)
        enemigo_v0y = params.velocidad_enemigo * np.cos(angulo_enemigo_rad) * np.sin(azimut_enemigo_rad)
        enemigo_v0z = params.velocidad_enemigo * np.sin(angulo_enemigo_rad)

        enemigo_x0 = params.distancia_defensa
        enemigo_y0 = 0
        enemigo_z0 = params.altura_enemigo

        enemigo_x = enemigo_x0 + enemigo_v0x * t
        enemigo_y = enemigo_y0 + enemigo_v0y * t
        enemigo_z = enemigo_z0 + enemigo_v0z * t - 0.5 * params.constantes.GRAVEDAD * np.power(t, 2)

        enemigo_z = np.maximum(enemigo_z, 0)

        antimisil_v0x = params.velocidad_antimisil * np.cos(angulo_antimisil_rad) * np.cos(azimut_antimisil_rad)
        antimisil_v0y = params.velocidad_antimisil * np.cos(angulo_antimisil_rad) * np.sin(azimut_antimisil_rad)
        antimisil_v0z = params.velocidad_antimisil * np.sin(angulo_antimisil_rad)

        mascara_tiempo = t >= params.tiempo_inicio_antimisil
        delta_t = np.zeros_like(t)
        delta_t[mascara_tiempo] = t[mascara_tiempo] - params.tiempo_inicio_antimisil

        antimisil_x = antimisil_v0x * delta_t
        antimisil_y = antimisil_v0y * delta_t
        antimisil_z = antimisil_v0z * delta_t - 0.5 * params.constantes.GRAVEDAD * np.power(delta_t, 2)

        antimisil_z = np.maximum(antimisil_z, 0)

        return t, enemigo_x, enemigo_y, enemigo_z, antimisil_x, antimisil_y, antimisil_z

    @staticmethod
    def verificar_intercepcion(params: ParametrosSimulacion,
                               t: np.ndarray,
                               enemigo_x: np.ndarray,
                               enemigo_y: np.ndarray,
                               antimisil_x: np.ndarray,
                               antimisil_y: np.ndarray) -> Tuple[bool, float, float, Optional[float], Optional[float]]:
        """Verifica si ocurre una intercepción y devuelve los detalles"""
        distancias = np.sqrt(np.power(enemigo_x - antimisil_x, 2) + np.power(enemigo_y - antimisil_y, 2))

        indice_distancia_minima = np.argmin(distancias)
        distancia_minima = distancias[indice_distancia_minima]
        tiempo_intercepcion = t[indice_distancia_minima]

        intercepcion = distancia_minima <= params.constantes.MARGEN_ERROR

        indices_suelo_enemigo = np.where(enemigo_y <= 0)[0]

        if len(indices_suelo_enemigo) > 0:
            tiempo_suelo_enemigo = t[indices_suelo_enemigo[0]]
            antes_impacto = tiempo_intercepcion < tiempo_suelo_enemigo
        else:
            antes_impacto = True

        if intercepcion and antes_impacto:
            interceptar_x = enemigo_x[indice_distancia_minima]
            interceptar_y = enemigo_y[indice_distancia_minima]
        else:
            interceptar_x = None
            interceptar_y = None

        return intercepcion and antes_impacto, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y

    @staticmethod
    def verificar_intercepcion_3d(params: ParametrosSimulacion,
                                  t: np.ndarray,
                                  enemigo_x: np.ndarray,
                                  enemigo_y: np.ndarray,
                                  enemigo_z: np.ndarray,
                                  antimisil_x: np.ndarray,
                                  antimisil_y: np.ndarray,
                                  antimisil_z: np.ndarray) -> Tuple[bool, float, float, Optional[float], Optional[float], Optional[float]]:
        """Verifica si ocurre una intercepción en 3D y devuelve los detalles"""
        distancias = np.sqrt(np.power(enemigo_x - antimisil_x, 2) +
                             np.power(enemigo_y - antimisil_y, 2) +
                             np.power(enemigo_z - antimisil_z, 2))

        indice_distancia_minima = np.argmin(distancias)
        distancia_minima = distancias[indice_distancia_minima]
        tiempo_intercepcion = t[indice_distancia_minima]

        intercepcion = distancia_minima <= params.constantes.MARGEN_ERROR

        indices_suelo_enemigo = np.where(enemigo_z <= 0)[0]

        if len(indices_suelo_enemigo) > 0:
            tiempo_suelo_enemigo = t[indices_suelo_enemigo[0]]
            antes_impacto = tiempo_intercepcion < tiempo_suelo_enemigo
        else:
            antes_impacto = True

        if intercepcion and antes_impacto:
            interceptar_x = enemigo_x[indice_distancia_minima]
            interceptar_y = enemigo_y[indice_distancia_minima]
            interceptar_z = enemigo_z[indice_distancia_minima]
        else:
            interceptar_x = None
            interceptar_y = None
            interceptar_z = None

        return intercepcion and antes_impacto, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y, interceptar_z

    @staticmethod
    def calcular_parametros_optimos(params: ParametrosSimulacion) -> Tuple[float, float]:
        """Calcula los parámetros óptimos para la intercepción"""
        angulo_enemigo_rad = np.radians(params.angulo_enemigo)
        enemigo_v0y = params.velocidad_enemigo * np.sin(angulo_enemigo_rad)
        enemigo_v0x = params.velocidad_enemigo * np.cos(angulo_enemigo_rad)

        a = 0.5 * params.constantes.GRAVEDAD
        b = -enemigo_v0y
        c = -params.altura_enemigo

        discriminante = b**2 - 4*a*c
        if discriminante < 0:
            return 45.0, 0.0

        tiempo_enemigo_suelo = (-b + math.sqrt(discriminante)) / (2*a)

        impacto_enemigo_x = params.distancia_defensa + enemigo_v0x * tiempo_enemigo_suelo

        distancia_intercepcion = impacto_enemigo_x / 2

        tiempo_intercepcion = distancia_intercepcion / (params.velocidad_antimisil * 0.7)

        pos_enemigo_x = params.distancia_defensa + enemigo_v0x * tiempo_intercepcion
        pos_enemigo_y = params.altura_enemigo + enemigo_v0y * tiempo_intercepcion - 0.5 * params.constantes.GRAVEDAD * tiempo_intercepcion**2

        delta_x = pos_enemigo_x
        delta_y = pos_enemigo_y

        angulo_optimo = 45.0
        tiempo_optimo = 0.0

        for _ in range(5):
            angulo_rad = math.atan2(delta_y, delta_x)
            angulo_rad += 0.1

            angulo_optimo = math.degrees(angulo_rad)
            angulo_optimo = max(0, min(90, angulo_optimo))

            v0x = params.velocidad_antimisil * math.cos(math.radians(angulo_optimo))
            tiempo_vuelo = delta_x / v0x if v0x > 0 else 0

            tiempo_optimo = max(0, tiempo_intercepcion - tiempo_vuelo)

            nuevo_tiempo_intercepcion = tiempo_optimo + tiempo_vuelo
            pos_enemigo_x = params.distancia_defensa + enemigo_v0x * nuevo_tiempo_intercepcion
            pos_enemigo_y = params.altura_enemigo + enemigo_v0y * nuevo_tiempo_intercepcion - 0.5 * params.constantes.GRAVEDAD * nuevo_tiempo_intercepcion**2

            delta_x = pos_enemigo_x
            delta_y = pos_enemigo_y

        if params.clima.condicion != CondicionClima.DESPEJADO:
            efecto_viento = params.clima.velocidad_viento / 100.0
            angulo_viento = math.radians(params.clima.direccion_viento)

            compensacion_viento = efecto_viento * math.sin(angulo_viento - math.radians(angulo_optimo))
            angulo_optimo += compensacion_viento * 5

            ajuste_tiempo = efecto_viento * 0.2
            tiempo_optimo = max(0, tiempo_optimo - ajuste_tiempo)

        return angulo_optimo, tiempo_optimo

class MotorSimulacion:
    """Motor de simulación para el sistema de defensa antimisiles"""

    def __init__(self):
        self.params = ParametrosSimulacion()
        self.resultados = None
        self.estado = EstadoSimulacion.INICIAL
        self._en_ejecucion = False
        self._ultimo_tiempo_simulacion = 0
        self._trayectorias_cacheadas = None

    def inicializar(self, params: ParametrosSimulacion):
        """Inicializa el motor con los parámetros dados"""
        self.params = params
        self.estado = EstadoSimulacion.INICIAL
        self._trayectorias_cacheadas = None

    def ejecutar_simulacion(self) -> ResultadosSimulacion:
        """Ejecuta la simulación completa y devuelve los resultados"""
        self.estado = EstadoSimulacion.EJECUTANDO
        self._en_ejecucion = True
        tiempo_inicio = time.time()

        try:
            if self.params.modo_vista == ModoVista.MODO_2D:
                trayectorias = self._calcular_trayectorias_2d()
                resultado_intercepcion = self._verificar_intercepcion_2d(trayectorias)
                self._trayectorias_cacheadas = trayectorias
            else:
                trayectorias = self._calcular_trayectorias_3d()
                resultado_intercepcion = self._verificar_intercepcion_3d(trayectorias)
                self._trayectorias_cacheadas = trayectorias

            self.resultados = self._crear_resultados_simulacion(resultado_intercepcion, trayectorias)
            self.estado = EstadoSimulacion.COMPLETADO
            return self.resultados

        except Exception as e:
            self.estado = EstadoSimulacion.FALLIDO
            print(f"Error en la simulación: {str(e)}")
            raise
        finally:
            self._en_ejecucion = False
            self._ultimo_tiempo_simulacion = time.time() - tiempo_inicio

    def _calcular_trayectorias_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcula las trayectorias para ambos misiles en 2D"""
        return CalculadorTrayectorias.calcular_trayectorias(self.params)

    def _calcular_trayectorias_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcula las trayectorias para ambos misiles en 3D"""
        return CalculadorTrayectorias.calcular_trayectorias_3d(self.params)

    def _verificar_intercepcion_2d(self, trayectorias: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[bool, float, float, Optional[float], Optional[float]]:
        """Verifica si ocurre una intercepción y devuelve los detalles"""
        return CalculadorTrayectorias.verificar_intercepcion(self.params, *trayectorias)

    def _verificar_intercepcion_3d(self, trayectorias: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[bool, float, float, Optional[float], Optional[float], Optional[float]]:
        """Verifica si ocurre una intercepción en 3D y devuelve los detalles"""
        return CalculadorTrayectorias.verificar_intercepcion_3d(self.params, *trayectorias)

    def _crear_resultados_simulacion(self, resultado_intercepcion: Tuple[bool, float, float, Optional[float], Optional[float]], trayectorias: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> ResultadosSimulacion:
        """Crea el objeto de resultados de la simulación"""
        t, enemigo_x, enemigo_y, antimisil_x, antimisil_y = trayectorias
        exito, tiempo_intercepcion, distancia_minima, interceptar_x, interceptar_y = resultado_intercepcion

        indices_suelo_enemigo = np.where(enemigo_y <= 0)[0]
        if len(indices_suelo_enemigo) > 0:
            tiempo_impacto_enemigo = t[indices_suelo_enemigo[0]]
            impacto_enemigo_x = enemigo_x[indices_suelo_enemigo[0]]
            coords_impacto_enemigo = (impacto_enemigo_x, 0, None)
        else:
            tiempo_impacto_enemigo = None
            coords_impacto_enemigo = None

        if exito:
            indice_tiempo = np.where(t >= tiempo_intercepcion)[0][0]
            trayectoria_misil_defensa = np.sum(np.sqrt(
                np.diff(antimisil_x[:indice_tiempo+1])**2 +
                np.diff(antimisil_y[:indice_tiempo+1])**2
            ))
            trayectoria_misil_enemigo = np.sum(np.sqrt(
                np.diff(enemigo_x[:indice_tiempo+1])**2 +
                np.diff(enemigo_y[:indice_tiempo+1])**2
            ))
        else:
            trayectoria_misil_defensa = np.sum(np.sqrt(
                np.diff(antimisil_x)**2 +
                np.diff(antimisil_y)**2
            ))
            trayectoria_misil_enemigo = np.sum(np.sqrt(
                np.diff(enemigo_x)**2 +
                np.diff(enemigo_y)**2
            ))

        tiempo_respuesta_defensa = self.params.tiempo_inicio_antimisil
        tasa_consumo_combustible = self.params.misil_defensa.consumo_combustible
        combustible_consumido = tasa_consumo_combustible * (tiempo_intercepcion - self.params.tiempo_inicio_antimisil) if exito else 0

        eficiencia_base = 100 if exito else 0
        penalizacion_tiempo = max(0, tiempo_intercepcion * 2)
        penalizacion_combustible = max(0, combustible_consumido / 10)
        puntuacion_eficiencia = max(0, eficiencia_base - penalizacion_tiempo - penalizacion_combustible)

        probabilidad_exito = 1.0 if distancia_minima <= self.params.constantes.MARGEN_ERROR/2 else \
                             max(0, 1 - (distancia_minima - self.params.constantes.MARGEN_ERROR/2) / self.params.constantes.MARGEN_ERROR)

        bajas_evitadas = None
        dano_evitado = None
        if exito:
            rendimiento_explosivo = self.params.misil_enemigo.rendimiento_explosivo
            bajas_evitadas = int(rendimiento_explosivo * 0.5)
            dano_evitado = rendimiento_explosivo * 1000

        return ResultadosSimulacion(
            exito=exito,
            tiempo_intercepcion=float(tiempo_intercepcion),
            distancia_minima=float(distancia_minima),
            coords_intercepcion=(interceptar_x, interceptar_y, None) if interceptar_x is not None else None,
            coords_impacto_enemigo=coords_impacto_enemigo,
            tiempo_impacto_enemigo=float(tiempo_impacto_enemigo) if tiempo_impacto_enemigo is not None else None,
            tiempo_respuesta_defensa=float(tiempo_respuesta_defensa),
            longitud_trayectoria_misil_defensa=float(trayectoria_misil_defensa),
            longitud_trayectoria_misil_enemigo=float(trayectoria_misil_enemigo),
            combustible_consumido=float(combustible_consumido),
            puntuacion_eficiencia=float(puntuacion_eficiencia),
            probabilidad_exito=float(probabilidad_exito),
            bajas_evitadas=bajas_evitadas,
            dano_evitado=dano_evitado
        )

    def obtener_trayectorias_cacheadas(self):
        """Devuelve las trayectorias calculadas en la última ejecución"""
        return self._trayectorias_cacheadas

    def optimizar_parametros(self) -> Tuple[float, float]:
        """Calcula los parámetros óptimos para la intercepción"""
        return CalculadorTrayectorias.calcular_parametros_optimos(self.params)

    def obtener_tiempo_ejecucion(self) -> float:
        """Devuelve el tiempo de ejecución de la última simulación"""
        return self._ultimo_tiempo_simulacion

    def esta_ejecutando(self) -> bool:
        """Indica si la simulación está en ejecución"""
        return self._en_ejecucion

    def cancelar(self):
        """Cancela la simulación en curso"""
        self._en_ejecucion = False
        self.estado = EstadoSimulacion.INICIAL
