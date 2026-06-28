import sys
import yaml
import math
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from core.ros_bridge import RobotBridge
from core.kinematics import transform_point_math

# ========== CONFIGURACIÓN GLOBAL ==========
CONFIG_FILE = "config/robot_config.yaml"
ROSBRIDGE_IP = "10.146.45.93" 
MAX_DIST = 1.9  

# SELECCIÓN DE SENSOR DE MAPEO
MAPPING_SENSOR_IDX = 0  # Índice del sensor frontal elegido para actualizar el mapa probabilístico

try:
    with open(CONFIG_FILE, 'r') as f:
        config_yaml = yaml.safe_load(f)
    ROBOT_LEN = config_yaml['robot']['length']
    ROBOT_WID = config_yaml['robot']['width']
    SENSORS = config_yaml['sensors']
except Exception as e:
    print(f"Error crítico cargando {CONFIG_FILE}: {e}")
    sys.exit(1)

# ========== PARÁMETROS DEL MAPA TERNARIO MUTABLE ==========
RESOLUTION_CM = 1.0  # Cada celda = 10mm (1cm)
MAX_EVIDENCE = 2.0   # Saturación máxima para permitir que el mapa reaccione rápido a cambios

# Parámetros Vacío Seguro (Clasificación 2)
SIGMA_EMPTY = 2.0    
MAG_EMPTY = 0.25      
THRES_EMPTY = 0.3    

# Parámetros Obstáculo/Muro (Clasificación 3)
SIGMA_OBS = 1.0      
MAG_OBS = 0.5       
THRES_OBS = 0.3      

empty_evidence_grid = {}
obstacle_evidence_grid = {}

prev_pose = None
prev_scan = None

bridge = RobotBridge(ip=ROSBRIDGE_IP)

def precompute_gaussian_kernel(sigma, magnitude, radius_cells):
    kernel = []
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            dist_sq = dx**2 + dy**2
            weight = magnitude * math.exp(-dist_sq / (2 * (sigma**2)))
            if weight > 0.05:  
                kernel.append((dx, dy, weight))
    return kernel

EMPTY_KERNEL = precompute_gaussian_kernel(SIGMA_EMPTY, MAG_EMPTY, radius_cells=1)
OBS_KERNEL = precompute_gaussian_kernel(SIGMA_OBS, MAG_OBS, radius_cells=2)

def get_bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

def main():
    print(f"Iniciando Mapeo Asilado con Sensor ID {MAPPING_SENSOR_IDX} (Visualización Total)...")
    bridge.connect()
    
    if not bridge.connected:
        print("❌ Fallo crítico de conexión.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    
    info_str = (
        f"MAPA MUTABLE AISLADO:\n"
        f" Sensor de Mapeo: ID {MAPPING_SENSOR_IDX}\n"
        f" Sensores de Gráfica: Todos (6)\n"
        f" Max Saturación: {MAX_EVIDENCE}\n\n"
        f"█ Vacío (Celeste):\n"
        f"  σ={int(SIGMA_EMPTY*10)}mm, Mag={MAG_EMPTY}\n\n"
        f"█ Muro (Gris):\n"
        f"  σ={int(SIGMA_OBS*10)}mm, Mag={MAG_OBS}"
    )
    ui_text = ax.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=9, 
                      verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    def update_frame(frame):
        global prev_pose, prev_scan, empty_evidence_grid, obstacle_evidence_grid
        
        ax.clear()
        ax.set_xlim(-4, 4) 
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)
        ax.add_artist(ui_text)

        pose, scan = bridge.get_state()
        if pose is None or scan is None:
            return

        if prev_pose is not None and prev_scan is not None:
            dist_moved = math.hypot(pose['x'] - prev_pose['x'], pose['y'] - prev_pose['y'])
            diff_theta = math.atan2(math.sin(pose['theta'] - prev_pose['theta']), 
                                    math.cos(pose['theta'] - prev_pose['theta']))
            sub_steps = max(1, int(max(dist_moved / 0.02, abs(diff_theta) / np.deg2rad(2.5))))
            sub_steps = min(sub_steps, 10)  
        else:
            sub_steps = 1

        # 1. BARRIDO DE VACÍO INTERPOLADO (RESTRINGIDO AL SENSOR SELECCIONADO)
        for step in range(1, sub_steps + 1):
            t = step / sub_steps
            
            if prev_pose is not None:
                interp_x = prev_pose['x'] + t * (pose['x'] - prev_pose['x'])
                interp_y = prev_pose['y'] + t * (pose['y'] - prev_pose['y'])
                interp_theta = prev_pose['theta'] + t * math.atan2(math.sin(pose['theta'] - prev_pose['theta']), 
                                                                   math.cos(pose['theta'] - prev_pose['theta']))
            else:
                interp_x, interp_y, interp_theta = pose['x'], pose['y'], pose['theta']

            cos_t = math.cos(interp_theta)
            sin_t = math.sin(interp_theta)

            # Extraemos la distancia correspondiente únicamente al sensor elegido
            if MAPPING_SENSOR_IDX < len(scan):
                dist = scan[MAPPING_SENSOR_IDX]
                if prev_scan is not None and prev_scan[MAPPING_SENSOR_IDX] is not None and dist is not None:
                    interp_dist = prev_scan[MAPPING_SENSOR_IDX] + t * (dist - prev_scan[MAPPING_SENSOR_IDX])
                else:
                    interp_dist = dist

                if interp_dist is not None and 0.02 < interp_dist < MAX_DIST:
                    sx_w = interp_x + SENSORS[MAPPING_SENSOR_IDX]['x'] * cos_t - SENSORS[MAPPING_SENSOR_IDX]['y'] * sin_t
                    sy_w = interp_y + SENSORS[MAPPING_SENSOR_IDX]['x'] * sin_t + SENSORS[MAPPING_SENSOR_IDX]['y'] * cos_t
                    
                    px_w = sx_w + interp_dist * math.cos(interp_theta + SENSORS[MAPPING_SENSOR_IDX]['theta'])
                    py_w = sy_w + interp_dist * math.sin(interp_theta + SENSORS[MAPPING_SENSOR_IDX]['theta'])

                    g_sx, g_sy = int(round(sx_w * 100 / RESOLUTION_CM)), int(round(sy_w * 100 / RESOLUTION_CM))
                    g_cx, g_cy = int(round(px_w * 100 / RESOLUTION_CM)), int(round(py_w * 100 / RESOLUTION_CM))

                    ray_cells = get_bresenham_line(g_sx, g_sy, g_cx, g_cy)
                    for rx, ry in ray_cells:
                        for kdx, kdy, kw in EMPTY_KERNEL:
                            cx_k, cy_k = rx + kdx, ry + kdy
                            
                            empty_evidence_grid[(cx_k, cy_k)] = min(MAX_EVIDENCE, empty_evidence_grid.get((cx_k, cy_k), 0.0) + kw)
                            if obstacle_evidence_grid.get((cx_k, cy_k), 0.0) > 0:
                                obstacle_evidence_grid[(cx_k, cy_k)] = max(0.0, obstacle_evidence_grid[(cx_k, cy_k)] - kw)

        # 2. PROCESAMIENTO ACTUAL (Mapeo de Obstáculo para ID seleccionado + Gráfica completa para los 6)
        cos_f = math.cos(pose['theta'])
        sin_f = math.sin(pose['theta'])
        curr_pts = []
        laser_segments = []

        for i, dist in enumerate(scan):
            if dist is not None and 0.02 < dist < MAX_DIST:
                sx_w = pose['x'] + SENSORS[i]['x'] * cos_f - SENSORS[i]['y'] * sin_f
                sy_w = pose['y'] + SENSORS[i]['x'] * sin_f + SENSORS[i]['y'] * cos_f
                px_w, py_w = transform_point_math(pose, SENSORS[i], dist)
                
                # RECOPILACIÓN GRÁFICA TOTAL: Se añaden todos los haces a las estructuras visuales
                curr_pts.append([px_w, py_w])
                laser_segments.append(([sx_w, px_w], [sy_w, py_w]))

                # FILTRO DE ACTUALIZACIÓN: Solo el sensor elegido inyecta y erosiona evidencia de obstáculos
                if i == MAPPING_SENSOR_IDX:
                    g_px, g_py = int(round(px_w * 100 / RESOLUTION_CM)), int(round(py_w * 100 / RESOLUTION_CM))

                    for kdx, kdy, kw in OBS_KERNEL:
                        ox_k, oy_k = g_px + kdx, g_py + kdy
                        
                        obstacle_evidence_grid[(ox_k, oy_k)] = min(MAX_EVIDENCE, obstacle_evidence_grid.get((ox_k, oy_k), 0.0) + kw)
                        if empty_evidence_grid.get((ox_k, oy_k), 0.0) > 0:
                            empty_evidence_grid[(ox_k, oy_k)] = max(0.0, empty_evidence_grid[(ox_k, oy_k)] - (kw * 1.5))

        prev_pose = pose
        prev_scan = scan

        # 3. RENDERIZADO DEL MAPA PROBABILÍSTICO (Generado únicamente por el sensor activo)
        celeste_list = [
            [k[0] * RESOLUTION_CM / 100.0, k[1] * RESOLUTION_CM / 100.0]
            for k, v in empty_evidence_grid.items() if v >= THRES_EMPTY and obstacle_evidence_grid.get(k, 0.0) < THRES_OBS
        ]
        
        gris_list = [
            [k[0] * RESOLUTION_CM / 100.0, k[1] * RESOLUTION_CM / 100.0]
            for k, v in obstacle_evidence_grid.items() if v >= THRES_OBS
        ]

        if celeste_list:
            arr_celeste = np.array(celeste_list)
            ax.scatter(arr_celeste[:, 0], arr_celeste[:, 1], c='#d6eaf8', s=6, marker='s', edgecolors='none', label='Vacío (Mutable)')

        if gris_list:
            arr_gris = np.array(gris_list)
            ax.scatter(arr_gris[:, 0], arr_gris[:, 1], c='#707b7c', s=7, marker='s', edgecolors='none', label='Muro (Mutable)')

        # Dibujar haces de luz de los 6 sensores en la interfaz gráfica
        for idx, (x_seg, y_seg) in enumerate(laser_segments):
            xlbl = "Haces ToF Activos (6)" if idx == 0 else "" 
            ax.plot(x_seg, y_seg, color='red', linewidth=1.0, alpha=0.25, label=xlbl)

        # Contorno físico del robot
        hx, hy = ROBOT_LEN / 2.0, ROBOT_WID / 2.0
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
        world_corners = []
        for cx, cy in corners:
            wx = pose['x'] + cx * cos_f - cy * sin_f
            wy = pose['y'] + cx * sin_f + cy * cos_f
            world_corners.append((wx, wy))
        ax.plot(np.array(world_corners)[:, 0], np.array(world_corners)[:, 1], color='#1f618d', linewidth=2, label="Robot")
        ax.plot(pose['x'], pose['y'], 'ko', markersize=4)

        # Ecos ToF instantáneos reflejados de todos los sensores
        if curr_pts:
            pts_curr = np.array(curr_pts)
            ax.scatter(pts_curr[:, 0], pts_curr[:, 1], c='#e74c3c', s=10, zorder=5, label='Ecos ToF Totales')

        ax.legend(loc='lower right', fontsize='8')

    ani = animation.FuncAnimation(fig, update_frame, interval=100, blit=False, cache_frame_data=False)
    plt.show()

    bridge.disconnect()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nCerrando comando central...")