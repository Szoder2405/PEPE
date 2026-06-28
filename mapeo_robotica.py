import cv2
import numpy as np
import cv2.aruco as aruco

class DetectorArenaCamara:
    def __init__(self, camara_id=0):
        # Inicializar cámara
        self.cap = cv2.VideoCapture(camara_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"ERROR: No se pudo abrir la cámara con ID {camara_id}. Reintentando con ID 0.")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        if not self.cap.isOpened():
            print("ERROR CRÍTICO: No se pudo abrir ninguna cámara.")
            return

        # --- CONFIGURACIÓN ARUCO ULTRA-ROBUSTA (ANTI-FALSOS POSITIVOS) ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        
        # Parámetros para evitar falsas detecciones en el fondo vacío
        self.aruco_params.minMarkerPerimeterRate = 0.015  # Elevado ligeramente para ignorar ruido diminuto del suelo
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        
        # Control estricto de error para evitar que el ruido se confunda con un ID válido
        self.aruco_params.errorCorrectionRate = 0.2  # Exigencia alta de bits correctos
        
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 11
        self.aruco_params.adaptiveThreshWinSizeStep = 2
        
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 3
        
        self.detector_aruco = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Preprocesador balanceado para no generar texturas fantasma
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))

        # Memoria y filtro pasa-bajas para el robot
        self.robot_x = None
        self.robot_y = None
        self.robot_angulo = 0
        self.ultima_caja = None
        self.alpha = 0.60            
        self.frames_perdido = 0
        self.MAX_FRAMES_MEMORIA = 25  # Alta persistencia para tolerar parpadeos largos del ArUco

        # --- RANGOS HSV UNIFICADOS ---
        self.rojo_bajo1 = np.array([0, 60, 30]);    self.rojo_alto1 = np.array([15, 255, 255])
        self.rojo_bajo2 = np.array([155, 60, 30]);  self.rojo_alto2 = np.array([180, 255, 255])
        self.azul_bajo = np.array([90, 30, 40]);    self.azul_alto = np.array([135, 255, 255])
        self.verde_bajo = np.array([35, 25, 30]);   self.verde_alto = np.array([85, 255, 255])
        self.negro_bajo = np.array([0, 0, 0]);      self.negro_alto = np.array([180, 255, 65])

        # --- KERNELS DE PROCESAMIENTO ---
        self.kernel_ruido = np.ones((3,3), np.uint8)
        self.kernel_cierre = np.ones((15,15), np.uint8)
        self.kernel_sombras = np.ones((9,9), np.uint8)
        self.kernel_obst = np.ones((5,5), np.uint8)
        # Kernel ampliado: Protege un área mayor alrededor del robot para absorber el chasis físico completo
        self.kernel_exclusion_aruco = np.ones((25,25), np.uint8) 

    def obtener_mascaras_limpias(self, hsv):
        m_r = cv2.bitwise_or(cv2.inRange(hsv, self.rojo_bajo1, self.rojo_alto1),
                             cv2.inRange(hsv, self.rojo_bajo2, self.rojo_alto2))
        m_a = cv2.inRange(hsv, self.azul_bajo, self.azul_alto)
        m_v = cv2.inRange(hsv, self.verde_bajo, self.verde_alto)

        m_r = cv2.morphologyEx(m_r, cv2.MORPH_OPEN, self.kernel_ruido)
        m_a = cv2.morphologyEx(m_a, cv2.MORPH_OPEN, self.kernel_ruido)
        m_v = cv2.morphologyEx(m_v, cv2.MORPH_OPEN, self.kernel_ruido)

        return m_r, m_a, m_v

    def detectar_robot(self, gray, frame, mapa):
        """Detecta el robot con super-resolución y mantiene una máscara de exclusión persistente."""
        gray_enhanced = self.clahe.apply(gray)
        
        # Escalado digital optimizado a 2X para mitigar aliasing y falsos positivos
        gray_resized = cv2.resize(gray_enhanced, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray_processed = cv2.GaussianBlur(gray_resized, (3, 3), 0)

        corners, ids, _ = self.detector_aruco.detectMarkers(gray_processed)
        mascara_aruco = np.zeros(gray.shape, dtype=np.uint8)

        # Validación estricta: Que existan IDs y que pertenezcan al rango esperado (ej: ID menor a 50)
        id_valido = False
        if ids is not None and len(ids) > 0:
            if ids[0][0] < 50:  # Filtro de seguridad para descartar decodificaciones fantasmas de ruido
                id_valido = True

        if id_valido:
            c = corners[0][0] / 2.0  # Retornar escala al espacio 360p
            
            cx_medido = int(np.mean(c[:, 0]))
            cy_medido = int(np.mean(c[:, 1]))

            p1 = (c[0] + c[1]) / 2
            ang_medido = np.degrees(np.arctan2(p1[1] - cy_medido, p1[0] - cx_medido))

            if self.robot_x is None:
                self.robot_x = cx_medido
                self.robot_y = cy_medido
                self.robot_angulo = ang_medido
            else:
                self.robot_x = int(self.alpha * cx_medido + (1 - self.alpha) * self.robot_x)
                self.robot_y = int(self.alpha * cy_medido + (1 - self.alpha) * self.robot_y)
                diff = (ang_medido - self.robot_angulo + 180) % 360 - 180
                self.robot_angulo += self.alpha * diff

            self.frames_perdido = 0
            self.ultima_caja = c
            
            pts = c.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        else:
            # Si se pierde el ArUco temporalmente, aumentamos el contador de fallos
            if self.robot_x is not None:
                self.frames_perdido += 1
                if self.frames_perdido > self.MAX_FRAMES_MEMORIA:
                    # Solo después de muchos frames estables sin rastro se limpia el robot de la pantalla
                    self.robot_x = None
                    self.robot_y = None
                    self.ultima_caja = None

        # --- GENERACIÓN DE MÁSCARA PERSISTENTE ANTI-OBSTÁCULOS ---
        # Crucial: Aunque el ArUco parpadee y falle en este frame, si está en memoria, 
        # seguimos tapando su posición para que procesar_negro no lo convierta en obstáculo.
        if self.robot_x is not None:
            cx, cy = self.robot_x, self.robot_y
            ang = self.robot_angulo
            
            # Dibujar en el mapa virtual
            cv2.circle(mapa, (cx, cy), 14, (0, 255, 255), 2)
            cv2.line(mapa, (cx, cy),
                     (cx + int(25 * np.cos(np.radians(ang))), cy + int(25 * np.sin(np.radians(ang)))),
                     (0, 255, 255), 3)
            cv2.putText(mapa, "ROBOT", (cx - 22, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Generar el escudo protector en la máscara de negros
            if self.ultima_caja is not None:
                cv2.fillConvexPoly(mascara_aruco, self.ultima_caja.astype(int), 255)
                # Dilatación masiva para absorber el cuerpo completo del robot e impedir "falsos obstáculos"
                mascara_aruco = cv2.dilate(mascara_aruco, self.kernel_exclusion_aruco, iterations=1)
            else:
                cv2.circle(mascara_aruco, (cx, cy), 35, 255, -1)

        return mascara_aruco

    def procesar_color(self, mask_pura, nombre, frame, mapa, color_bgr):
        mask_cerrada = cv2.morphologyEx(mask_pura, cv2.MORPH_CLOSE, self.kernel_cierre)
        contornos, _ = cv2.findContours(mask_cerrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < 350:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi_pura = mask_pura[y:y+h, x:x+w]
            pixeles_reales = cv2.countNonZero(roi_pura)
            solidez_pura = pixeles_reales / (w * h + 1e-5)

            umbral_solidez = 0.60
            if nombre == "Azul":
                umbral_solidez = 0.78
            elif nombre == "Verde":
                relacion_aspecto = max(w, h) / min(w, h)
                if relacion_aspecto < 1.25:
                    umbral_solidez = 0.62   
                else:
                    umbral_solidez = 0.42   

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            if solidez_pura > umbral_solidez:
                rect = cv2.minAreaRect(cnt)
                box = np.array(cv2.boxPoints(rect), dtype=np.intp)
                cv2.drawContours(frame, [cnt], -1, color_bgr, 3)
                cv2.putText(frame, f"CUBO {nombre}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                cv2.drawContours(mapa, [box], 0, color_bgr, 2)
                cv2.putText(mapa, f"CUBO {nombre}", (cx-30, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
            else:
                rect = cv2.minAreaRect(cnt)
                box = np.array(cv2.boxPoints(rect), dtype=np.intp)
                cv2.drawContours(frame, [box], 0, color_bgr, 2)
                cv2.putText(frame, f"ZONA {nombre}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                offset = 14
                cv2.line(mapa, (cx-offset, cy-offset), (cx+offset, cy+offset), color_bgr, 2)
                cv2.line(mapa, (cx-offset, cy+offset), (cx+offset, cy-offset), color_bgr, 2)
                cv2.putText(mapa, f"ZONA {nombre}", (cx-30, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)

    def procesar_negro(self, hsv, frame, mapa, mascara_colores_dilatada, mascara_aruco):
        if not hasattr(self, 'pared_memoria'): self.pared_memoria = None
        if not hasattr(self, 'frames_perdidos_pared'): self.frames_perdidos_pared = 0

        mask = cv2.inRange(hsv, self.negro_bajo, self.negro_alto)
        mask = cv2.subtract(mask, mascara_colores_dilatada)
        mask = cv2.subtract(mask, mascara_aruco)  # Aquí se resta el robot (sea real o en memoria transitoria)

        kernel_soldadura = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_soldadura)
        mask = cv2.medianBlur(mask, 5)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_validos = [cnt for cnt in contornos if cv2.contourArea(cnt) > 40]
        
        hay_pared_este_frame = False
        pared_cnt = None

        if contornos_validos:
            contornos_validos = sorted(contornos_validos, key=cv2.contourArea, reverse=True)
            posible_pared = contornos_validos[0]
            area_pared = cv2.contourArea(posible_pared)
            umbral_exigido = 250 if self.pared_memoria is not None else 500

            if area_pared > umbral_exigido:
                pared_cnt = posible_pared
                self.pared_memoria = pared_cnt
                self.frames_perdidos_pared = 0
                hay_pared_este_frame = True

        if not hay_pared_este_frame and self.pared_memoria is not None:
            if self.frames_perdidos_pared < 6:
                pared_cnt = self.pared_memoria
                self.frames_perdidos_pared += 1
                hay_pared_este_frame = True
            else:
                self.pared_memoria = None

        if hay_pared_este_frame and pared_cnt is not None:
            rect_p = cv2.minAreaRect(pared_cnt)
            (xp, yp), _, _ = rect_p
            cv2.drawContours(frame, [pared_cnt], -1, (255, 0, 255), 2)
            cv2.putText(frame, "PARED", (int(xp)-20, int(yp)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.drawContours(mapa, [pared_cnt], -1, (255, 0, 255), 2)
            cv2.putText(mapa, "PARED", (int(xp)-20, int(yp)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        if hay_pared_este_frame and contornos_validos and self.frames_perdidos_pared == 0:
            obstaculos_candidatos = contornos_validos[1:]
        else:
            obstaculos_candidatos = contornos_validos

        alto_frame, ancho_frame = mask.shape

        for cnt in obstaculos_candidatos:
            if hay_pared_este_frame and cnt is pared_cnt:
                continue

            area = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
            box = np.array(cv2.boxPoints(rect), dtype=np.intp)
            (x, y), (w, h), _ = rect
            ancho, alto = max(w, h), min(w, h)

            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            solidez = area / (area_hull + 1e-5)
            relacion_aspecto = ancho / (alto + 1e-5)
            extent = area / (ancho * alto + 1e-5)

            min_dist_borde = min(x, y, ancho_frame - (x + w), alto_frame - (y + h))
            esta_aislado = min_dist_borde > 45  

            es_esquina_hueca = relacion_aspecto <= 2.2 and (extent < 0.68 or solidez < 0.78)
            es_pared_alargada = relacion_aspecto > 2.2

            if (es_pared_alargada or es_esquina_hueca or area > 1200) and not esta_aislado:
                cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)
                cv2.putText(frame, "PARED", (int(x)-20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                cv2.drawContours(mapa, [cnt], -1, (255, 0, 255), 2)
                cv2.putText(mapa, "PARED", (int(x)-20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            else:
                cv2.drawContours(frame, [box], 0, (40, 40, 40), 2)
                cv2.putText(frame, "OBST", (int(x)-15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
                cv2.drawContours(mapa, [box], 0, (120, 120, 120), 2)
                cv2.putText(mapa, "OBST", (int(x)-15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    def ejecutar(self):
        if not self.cap.isOpened():
            return
        print("Detección Optimizada: Solución al parpadeo, falsos obstáculos y robots fantasmas.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            mapa = np.zeros((360, 640, 3), dtype=np.uint8)   
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 1. Detectar robot con lógica anti-fantasmas y persistencia morfológica
            mascara_aruco = self.detectar_robot(gray, frame, mapa)

            # 2. Procesar colores
            m_r, m_a, m_v = self.obtener_mascaras_limpias(hsv)
            self.procesar_color(m_r, "Rojo", frame, mapa, (0, 0, 255))
            self.procesar_color(m_a, "Azul", frame, mapa, (255, 0, 0))
            self.procesar_color(m_v, "Verde", frame, mapa, (0, 255, 0))

            # 3. Máscara de colores dilatada para sombras
            mascara_colores = cv2.bitwise_or(m_r, cv2.bitwise_or(m_a, m_v))
            mascara_colores_dilatada = cv2.dilate(mascara_colores, self.kernel_sombras, iterations=1)

            # 4. Procesar negros utilizando la máscara protectora del robot
            self.procesar_negro(hsv, frame, mapa, mascara_colores_dilatada, mascara_aruco)

            # Mostrar resultados
            cv2.imshow("Deteccion de Objetos - Camara", frame)
            cv2.imshow("Mapa Limpio de la Arena", mapa)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DetectorArenaCamara(camara_id=1).ejecutar()