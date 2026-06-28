#!/usr/bin/env python3
"""
Visualizador de robot y LiDAR en Windows usando roslibpy y matplotlib
Recibe datos de ROS2 a través de rosbridge (WebSocket)
"""

import sys
import yaml
import math
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from roslibpy import Ros, Topic, Message

# ========== CONFIGURACIÓN ==========
ROSBRIDGE_IP = "10.146.45.93"   # <-- CAMBIAR a IP de tu Raspberry Pi
ROSBRIDGE_PORT = 9090
CONFIG_FILE = "config/robot_config.yaml"

# ========== CARGAR CONFIGURACIÓN ==========
try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    ROBOT_LEN = config['robot']['length']
    ROBOT_WID = config['robot']['width']
    SENSORS = config['sensors']
    print(f"Configuración cargada: Robot {ROBOT_LEN}m x {ROBOT_WID}m")
    print(f"Sensores: {len(SENSORS)}")
except Exception as e:
    print(f"Error cargando {CONFIG_FILE}: {e}")
    sys.exit(1)

# ========== DATOS GLOBALES ==========
robot_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
lidar_ranges = [0.0] * 6
data_lock = threading.Lock()
connected = False

# ========== FUNCIONES DE TRANSFORMACIÓN ==========
def transform_point(x, y, theta, px, py):
    """Transforma un punto local a coordenadas globales"""
    c = math.cos(theta)
    s = math.sin(theta)
    px_w = x + px * c - py * s
    py_w = y + px * s + py * c
    return px_w, py_w

def compute_world_points():
    """Calcula puntos LiDAR en coordenadas globales"""
    points = []
    with data_lock:
        ox = robot_pose["x"]
        oy = robot_pose["y"]
        otheta = robot_pose["theta"]
        ranges = lidar_ranges[:]

    for i, sensor in enumerate(SENSORS):
        if i < len(ranges):
            d = ranges[i]
            # Filtrar lecturas inválidas
            if d > 2.0 or d < 0.02:
                continue

            # Punto medido en el sistema del sensor
            px_local = d
            py_local = 0.0

            # Transformar a coordenadas del robot
            c_s = math.cos(sensor['theta'])
            s_s = math.sin(sensor['theta'])
            px_robot = sensor['x'] + px_local * c_s - py_local * s_s
            py_robot = sensor['y'] + px_local * s_s + py_local * c_s

            # Transformar a mundo
            px_w, py_w = transform_point(ox, oy, otheta, px_robot, py_robot)
            points.append((px_w, py_w))

    return points

# ========== CALLBACKS ROS ==========
def odom_callback(msg):
    with data_lock:
        robot_pose["x"] = msg['pose']['pose']['position']['x']
        robot_pose["y"] = msg['pose']['pose']['position']['y']
        q = msg['pose']['pose']['orientation']
        # Convertir quaternion a yaw (ángulo)
        siny_cosp = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
        cosy_cosp = 1.0 - 2.0 * (q['y'] * q['y'] + q['z'] * q['z'])
        robot_pose["theta"] = math.atan2(siny_cosp, cosy_cosp)

def scan_callback(msg):
    ranges = msg.get('ranges', [])
    with data_lock:
        for i in range(min(6, len(ranges))):
            val = ranges[i]
            if val == float('inf') or val > 2.0:
                lidar_ranges[i] = 2.5  # ignorar
            else:
                lidar_ranges[i] = val

# ========== CONEXIÓN A ROSBRIDGE ==========
def connect_ros():
    global connected
    client = Ros(ROSBRIDGE_IP, ROSBRIDGE_PORT)
    client.on_ready(lambda: print("✅ Conectado a ROSBridge"))
    client.run()
    connected = True
    return client

# ========== VISUALIZACIÓN ==========
def main():
    global connected

    print(f"Conectando a {ROSBRIDGE_IP}:{ROSBRIDGE_PORT}...")
    client = connect_ros()

    # Suscripciones
    odom_topic = Topic(client, '/odom', 'nav_msgs/Odometry')
    odom_topic.subscribe(odom_callback)

    scan_topic = Topic(client, '/scan', 'sensor_msgs/LaserScan')
    scan_topic.subscribe(scan_callback)

    time.sleep(1)  # esperar suscripciones

    if not connected:
        print("❌ No se pudo conectar a ROSBridge")
        return

    print("✅ Conectado. Esperando datos...")

    # Configurar matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Robot y sensores LiDAR", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    robot_poly = None
    lidar_scatter = None
    center_point = None

    def update_frame(frame):
        nonlocal robot_poly, lidar_scatter, center_point

        # Limpiar elementos previos
        if robot_poly:
            robot_poly.remove()
        if lidar_scatter:
            lidar_scatter.remove()
        if center_point:
            center_point.remove()

        # Dibujar robot (rectángulo)
        half_len = ROBOT_LEN / 2.0
        half_wid = ROBOT_WID / 2.0
        corners = np.array([
            [-half_len, -half_wid],
            [ half_len, -half_wid],
            [ half_len,  half_wid],
            [-half_len,  half_wid],
            [-half_len, -half_wid]
        ])

        with data_lock:
            ox = robot_pose["x"]
            oy = robot_pose["y"]
            otheta = robot_pose["theta"]

        world_corners = []
        for cx, cy in corners:
            wx, wy = transform_point(ox, oy, otheta, cx, cy)
            world_corners.append((wx, wy))
        world_corners = np.array(world_corners)
        robot_poly = ax.plot(world_corners[:, 0], world_corners[:, 1], 'b-', linewidth=2, label='Robot')[0]

        # Centro del robot
        center_point = ax.plot(ox, oy, 'ko', markersize=6)[0]

        # Puntos LiDAR
        world_points = compute_world_points()
        if world_points:
            pts = np.array(world_points)
            lidar_scatter = ax.scatter(pts[:, 0], pts[:, 1], c='red', s=10, alpha=0.7, label='Puntos LiDAR')
        else:
            lidar_scatter = ax.scatter([], [], c='red', s=10)

        # Leyenda (evitar duplicados)
        ax.legend(loc='upper right')

        return robot_poly, lidar_scatter, center_point

    ani = animation.FuncAnimation(fig, update_frame, interval=50, blit=False)
    plt.show()

    # Cierre limpio
    odom_topic.unsubscribe()
    scan_topic.unsubscribe()
    client.terminate()
    print("Desconectado")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nCerrando visualizador...")
    except Exception as e:
        print(f"Error: {e}")