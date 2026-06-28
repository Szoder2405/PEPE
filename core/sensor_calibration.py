import numpy as np
from scipy.optimize import minimize
from .kinematics import transform_point_math

def calibration_cost_function(flat_params, data_buffer, num_sensors):
    """
    Función de costo para Scipy. 
    flat_params es un array 1D: [x0, y0, theta0, x1, y1, theta1...]
    """
    all_points = []
    
    # Reconstruir la nube de puntos histórica con los parámetros propuestos
    for snapshot in data_buffer:
        pose = snapshot['pose']
        scan = snapshot['scan']
        
        for i in range(num_sensors):
            dist = scan[i]
            if 0.02 < dist < 2.0:
                idx = i * 3
                sensor_temp = {
                    'x': flat_params[idx],
                    'y': flat_params[idx+1],
                    'theta': flat_params[idx+2]
                }
                px, py = transform_point_math(pose, sensor_temp, dist)
                all_points.append([px, py])
                
    if not all_points:
        return float('inf')
        
    pts = np.array(all_points)
    
    # Minimizamos la varianza espacial (buscamos que los puntos se agrupen y no se dispersen)
    centroid = np.mean(pts, axis=0)
    variance = np.mean(np.sum((pts - centroid)**2, axis=1))
    
    return variance

def run_calibration_optimizer(data_buffer, current_config):
    print("\n[Optimizador] Procesando rotación. Ejecutando minimización...")
    
    initial_params = []
    bounds = []
    
    # Preparar el estado inicial y los límites lógicos para Scipy
    for s in current_config:
        initial_params.extend([s['x'], s['y'], s['theta']])
        # Límites: Permitir corregir hasta +/- 4cm en X,Y y +/- 15 grados en rotación
        bounds.extend([
            (s['x'] - 0.04, s['x'] + 0.04),
            (s['y'] - 0.04, s['y'] + 0.04),
            (s['theta'] - 0.26, s['theta'] + 0.26)
        ])
        
    initial_guess = np.array(initial_params)
    
    result = minimize(calibration_cost_function, 
                      initial_guess, 
                      args=(data_buffer, len(current_config)), 
                      method='L-BFGS-B', 
                      bounds=bounds)
                      
    if result.success:
        print("[Optimizador] ¡Convergencia lograda!")
        optimized_params = result.x
        
        # Formatear salida para actualizar el YAML
        new_config = []
        for i in range(len(current_config)):
            idx = i * 3
            new_config.append({
                'id': current_config[i].get('id', i),
                'x': float(optimized_params[idx]),
                'y': float(optimized_params[idx+1]),
                'theta': float(optimized_params[idx+2])
            })
        return new_config
    else:
        print("[Optimizador] Fallo en convergencia:", result.message)
        return current_config