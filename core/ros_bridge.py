import threading
import time
from roslibpy import Ros, Topic
from .kinematics import quaternion_to_yaw

class RobotBridge:
    def __init__(self, ip="10.146.45.93", port=9090):
        self.ip = ip
        self.port = port
        self.client = None
        self.connected = False
        self.data_lock = threading.Lock()
        
        # Estado interno
        self.pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.lidar_ranges = [2.5] * 6  # Inicializado fuera de rango
        
    def connect(self):
        self.client = Ros(self.ip, self.port)
        self.client.on_ready(self._on_ready)
        self.client.run()
        
        time.sleep(1.5) # Esperar a que los sockets conecten
        if self.connected:
            self.odom_topic = Topic(self.client, '/odom', 'nav_msgs/Odometry')
            self.odom_topic.subscribe(self._odom_callback)
            
            self.scan_topic = Topic(self.client, '/scan', 'sensor_msgs/LaserScan')
            self.scan_topic.subscribe(self._scan_callback)

    def _on_ready(self):
        self.connected = True
        print(f"✅ Conectado a ROSBridge en {self.ip}:{self.port}")

    def _odom_callback(self, msg):
        with self.data_lock:
            self.pose["x"] = msg['pose']['pose']['position']['x']
            self.pose["y"] = msg['pose']['pose']['position']['y']
            self.pose["theta"] = quaternion_to_yaw(msg['pose']['pose']['orientation'])

    def _scan_callback(self, msg):
        ranges = msg.get('ranges', [])
        with self.data_lock:
            for i in range(min(6, len(ranges))):
                val = ranges[i]
                # Filtrar valores infinitos o fuera del rango del VL53L0X
                if val == float('inf') or val > 2.0 or val < 0.02:
                    self.lidar_ranges[i] = 2.5
                else:
                    self.lidar_ranges[i] = val

    def get_state(self):
        """Retorna una copia segura del estado actual para la interfaz gráfica"""
        with self.data_lock:
            return self.pose.copy(), self.lidar_ranges[:]

    def disconnect(self):
        if self.connected:
            self.odom_topic.unsubscribe()
            self.scan_topic.unsubscribe()
            self.client.terminate()