import math
import numpy as np

def transform_point_math(robot_pose, sensor_params, measured_distance):
    """
    Transforma la lectura de un sensor a coordenadas globales.
    """
    ox, oy, otheta = robot_pose['x'], robot_pose['y'], robot_pose['theta']
    sx, sy, stheta = sensor_params['x'], sensor_params['y'], sensor_params['theta']
    
    # Transformar a coordenadas locales del robot
    px_robot = sx + measured_distance * math.cos(stheta)
    py_robot = sy + measured_distance * math.sin(stheta)
    
    # Transformar a coordenadas del mundo (mapa)
    px_world = ox + px_robot * math.cos(otheta) - py_robot * math.sin(otheta)
    py_world = oy + px_robot * math.sin(otheta) + py_robot * math.cos(otheta)
    
    return px_world, py_world

def quaternion_to_yaw(q):
    """Convierte un cuaternión de ROS2 a ángulo Yaw (Radianes)"""
    siny_cosp = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
    cosy_cosp = 1.0 - 2.0 * (q['y'] * q['y'] + q['z'] * q['z'])
    return math.atan2(siny_cosp, cosy_cosp)