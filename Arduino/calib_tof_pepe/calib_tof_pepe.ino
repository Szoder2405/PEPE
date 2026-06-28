#include <Wire.h>
#include <VL53L0X.h>

#define XSHUT_1 13
#define XSHUT_2 33
#define XSHUT_3 14
#define XSHUT_4 27
#define XSHUT_5 26
#define XSHUT_6 25

const int xshut_pins[6] = {XSHUT_1, XSHUT_2, XSHUT_3, XSHUT_4, XSHUT_5, XSHUT_6};
const uint8_t VL53L0X_ADDR[6] = {0x30, 0x31, 0x32, 0x33, 0x34, 0x35};

VL53L0X laser[6];
bool laser_ok[6] = {false};

// Almacenamiento de pares de calibración
const int MAX_SAMPLES = 10;
float real_dist[MAX_SAMPLES];        // distancias reales (mm) ingresadas
uint16_t raw_dist[6][MAX_SAMPLES];   // lecturas crudas de cada sensor
int sample_count = 0;

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(18, 19);
  Wire.setClock(50000);
  Serial.println("\n--- CALIBRACIÓN VL53L0X ---");
  Serial.println("Asegúrate de colocar un objeto a una distancia conocida.");

  // Inicializar sensores
  for (int i = 0; i < 6; i++) {
    pinMode(xshut_pins[i], OUTPUT);
    digitalWrite(xshut_pins[i], LOW);
  }
  delay(50);
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
      Serial.printf("Sensor %d -> OK\n", i);
    } else {
      laser_ok[i] = false;
      digitalWrite(xshut_pins[i], LOW);
      Serial.printf("Sensor %d -> FAILED\n", i);
    }
  }
  Serial.println("Ingresa la distancia real en mm (ej: 100) y presiona Enter.");
  Serial.println("Cuando tengas suficientes muestras, escribe 'calc' para ver resultados.");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() == 0) return;

    if (input.equalsIgnoreCase("calc")) {
      calculateAndPrint();
      return;
    }

    // Convertir a número
    float dist_mm = input.toFloat();
    if (dist_mm <= 0 || dist_mm > 2000) {
      Serial.println("Valor no válido. Introduce una distancia entre 1 y 2000 mm.");
      return;
    }

    if (sample_count >= MAX_SAMPLES) {
      Serial.println("Límite de muestras alcanzado. Escribe 'calc' para ver resultados.");
      return;
    }

    // Tomar lectura de todos los sensores
    real_dist[sample_count] = dist_mm;
    for (int i = 0; i < 6; i++) {
      if (laser_ok[i]) {
        uint16_t d = laser[i].readRangeContinuousMillimeters();
        raw_dist[i][sample_count] = d;
      } else {
        raw_dist[i][sample_count] = 0;
      }
    }
    Serial.printf("Muestra %d guardada (distancia %.1f mm)\n", sample_count + 1, dist_mm);
    sample_count++;
  }
}

void calculateAndPrint() {
  if (sample_count < 2) {
    Serial.println("Necesitas al menos 2 muestras para calcular calibración.");
    return;
  }

  Serial.println("\n--- RESULTADOS DE CALIBRACIÓN ---");
  for (int s = 0; s < 6; s++) {
    if (!laser_ok[s]) {
      Serial.printf("Sensor %d: no funcional, se omite.\n", s);
      continue;
    }

    // Regresión lineal simple: real = a * raw + b
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = sample_count;
    for (int i = 0; i < n; i++) {
      float x = raw_dist[s][i];
      float y = real_dist[i];
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_xx += x * x;
    }
    float denom = n * sum_xx - sum_x * sum_x;
    float a = (n * sum_xy - sum_x * sum_y) / denom;
    float b = (sum_y - a * sum_x) / n;

    // Imprimir coeficientes
    Serial.printf("Sensor %d: real = %.6f * raw + %.4f\n", s, a, b);
    // También mostramos el offset equivalente en mm para una lectura de 0 (aunque no sea físico)
    Serial.printf("   Offset aprox: %.2f mm, Scale: %.4f\n", b, a);
  }
  Serial.println("Copia estos coeficientes en tu código de producción.");
  Serial.println("Ejemplo de corrección: dist_corregida = a * dist_cruda + b;");
}