import json
from typing import Dict, Any
from models import ParametrosSimulacion

class Serializador:
    """Clase para serializar y deserializar objetos"""

    @staticmethod
    def serializar_a_json(objeto: Any) -> str:
        """Serializa un objeto a JSON"""
        if isinstance(objeto, ParametrosSimulacion):
            return json.dumps(objeto.a_diccionario())
        else:
            raise ValueError("Tipo de objeto no soportado para serialización")

    @staticmethod
    def deserializar_desde_json(json_str: str, tipo: type) -> Any:
        """Deserializa un objeto desde JSON"""
        if tipo == ParametrosSimulacion:
            data = json.loads(json_str)
            return ParametrosSimulacion.desde_diccionario(data)
        else:
            raise ValueError("Tipo de objeto no soportado para deserialización")

class Validador:
    """Clase para validar parámetros y entradas"""

    @staticmethod
    def validar_parametros_simulacion(params: ParametrosSimulacion) -> bool:
        """Valida los parámetros de simulación"""
        if params.altura_enemigo <= 0:
            return False
        if params.distancia_defensa <= 0:
            return False
        if params.velocidad_antimisil <= 0:
            return False
        if params.velocidad_enemigo <= 0:
            return False
        return True
