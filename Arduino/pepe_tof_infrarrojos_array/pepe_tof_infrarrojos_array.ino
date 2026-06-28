#include <Wire.h>
#include <VL53L0X.h>

// ==================== PINES VL53L0X (XSHUT) ====================
const int xshut_pins[6] = {13, 33, 14, 27, 26, 25};
const uint8_t VL53L0X_ADDR[6] = {0x30, 0x31, 0x32, 0x33, 0x34, 0x35};

VL53L0X laser[6];
bool laser_ok[6] = {false};

// Coeficientes de calibración (distancia)
float cal_a[6] = {0.873739, 0.907897, 0.907471, 0.919777, 1.0, 1.033659};
float cal_b[6] = {-60.3550, -45.3924, -47.8188, -22.8812, -64.64, -71.7347};

// Variables para control de frecuencia de envío de LiDAR
unsigned long last_send = 0;
unsigned long send_interval_ms = 50;   // 20 Hz

// ==================== PINES ELEVADOR (montacargas) ====================
#define LIFT_IN1  2   // D2 (GPIO2)
#define LIFT_IN2  4   // D4 (GPIO4)

volatile bool lift_up = false;
volatile bool lift_down = false;

// ==================== FUNCIONES ELEVADOR ====================
void liftStop() {
  digitalWrite(LIFT_IN1, LOW);
  digitalWrite(LIFT_IN2, LOW);
  lift_up = false;
  lift_down = false;
}

void liftUp() {
  digitalWrite(LIFT_IN1, HIGH);
  digitalWrite(LIFT_IN2, LOW);
  // Los flags se mantienen para que el loop repita la orden
}

void liftDown() {
  digitalWrite(LIFT_IN1, LOW);
  digitalWrite(LIFT_IN2, HIGH);
}

// ==================== CONFIGURACIÓN ====================
void setup() {
  Serial.begin(115200);
  delay(500);

  // Configurar pines del elevador
  pinMode(LIFT_IN1, OUTPUT);
  pinMode(LIFT_IN2, OUTPUT);
  liftStop();

  // I2C para sensores ToF
  Wire.begin(18, 19);
  Wire.setClock(10000);
  Serial.println("--- ESP32 ToF: iniciando 6 sensores ---");

  // Apagar todos los sensores
  for (int i = 0; i < 6; i++) {
    pinMode(xshut_pins[i], OUTPUT);
    digitalWrite(xshut_pins[i], LOW);
  }
  delay(50);

  // Inicializar uno por uno
  for (int i = 0; i < 6; i++) {
    digitalWrite(xshut_pins[i], HIGH);
    delay(150);
    if (laser[i].init(true)) {
      laser_ok[i] = true;
      laser[i].setTimeout(500);
      delay(100);
      laser[i].setAddress(VL53L0X_ADDR[i]);
      laser[i].setMeasurementTimingBudget(20000);
      laser[i].startContinuous();
      Serial.printf("Sensor %d OK\n", i);
    } else {
      laser_ok[i] = false;
      digitalWrite(xshut_pins[i], LOW);
      Serial.printf("Sensor %d FAIL\n", i);
    }
  }
  Serial.println("--- Enviando datos LiDAR cada 50 ms (20 Hz) ---");
  Serial.println("--- Comandos elevador: U,1 (subir), D,1 (bajar), S,0 (detener) ---");
}

// ==================== BUCLE PRINCIPAL ====================
void loop() {
  // Leer comandos serie (tanto para ToF como para elevador)
  while (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // Comandos para el ToF
    if (cmd == "X,1") {
      Serial.println("Reiniciando por comando...");
      delay(100);
      ESP.restart();
    }
    else if (cmd.startsWith("FREQ,")) {
      int new_interval = cmd.substring(5).toInt();
      if (new_interval >= 20 && new_interval <= 1000) {
        send_interval_ms = new_interval;
        Serial.printf("Intervalo de LiDAR cambiado a %d ms\n", send_interval_ms);
      } else {
        Serial.println("ERROR: intervalo 20-1000 ms");
      }
    }
    // Comandos para el elevador
    else if (cmd == "D,1") {
      liftStop();
      lift_up = true;
      lift_down = false;
      Serial.println("Elevador subiendo");
    }
    else if (cmd == "U,1") {
      liftStop();
      lift_up = false;
      lift_down = true;
      Serial.println("Elevador bajando");
    }
    else if (cmd == "S,0") {
      liftStop();
      Serial.println("Elevador detenido");
    }
  }

  // Control del elevador (no bloqueante)
  if (lift_up) liftUp();
  else if (lift_down) liftDown();
  else liftStop();

  // Envío periódico de datos LiDAR
  unsigned long now = millis();
  if (now - last_send >= send_interval_ms) {
    last_send = now;

    Serial.print("L");
    for (int i = 0; i < 6; i++) {
      Serial.print(",");
      if (laser_ok[i]) {
        uint16_t raw = laser[i].readRangeContinuousMillimeters();
        if (laser[i].timeoutOccurred() || raw > 8000) {
          Serial.print("8000");
        } else {
          float corrected = cal_a[i] * raw + cal_b[i];
          if (corrected < 0) corrected = 0;
          if (corrected > 2000) corrected = 2000;
          Serial.print((int)corrected);
        }
      } else {
        Serial.print("ERR");
      }
    }
    Serial.println();

    // Pequeña pausa para evitar saturación del buffer UART
    delay(1);
  }
}