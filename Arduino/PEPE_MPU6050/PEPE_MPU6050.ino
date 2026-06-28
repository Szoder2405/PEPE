#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_light.h>

// ==================== CONSTANTES ====================
#define BAUD_RATE           115200
#define ACCEL_WINDOW        20
#define GYRO_WINDOW         10
#define VEL_WINDOW          10

enum State {WAITING, CALIBRATING, MEASURING_NOISE, RUNNING};
State state = WAITING;

// ==================== OBJETOS Y VARIABLES ====================
MPU6050 mpu(Wire);
bool imu_calibrated = false;
unsigned long last_time = 0;

float ax_history[ACCEL_WINDOW] = {0};
float ay_history[ACCEL_WINDOW] = {0};
float wz_history[GYRO_WINDOW] = {0};
int accel_idx = 0, gyro_idx = 0, vel_idx = 0;

// Buffers para velocidad lineal
float vx_history[VEL_WINDOW] = {0};
float vy_history[VEL_WINDOW] = {0};

// Umbrales de ruido y medias de referencia (calculados en fase MEASURING_NOISE)
float threshold_ax = 0.5, threshold_ay = 0.5, threshold_wz = 0.5;
float threshold_vx = 0.5, threshold_vy = 0.5;
float ref_ax = 0, ref_ay = 0, ref_wz = 0;

// Variables para la fase de medición de ruido
unsigned long noise_start_time = 0;
const unsigned long NOISE_DURATION = 3000;
float max_abs_ax = 0, max_abs_ay = 0, max_abs_wz = 0;
float max_abs_vx = 0, max_abs_vy = 0;
float sum_ax = 0, sum_ay = 0, sum_wz = 0;
int noise_sample_count = 0;

// Variables de integración (redondeo a precisión real)
float vx = 0.0, vy = 0.0;
float px = 0.0, py = 0.0;
float theta = 0.0;

// Funciones para redondear a una precisión determinada
inline float roundTo(float val, int decimals) {
  float multiplier = powf(10.0, decimals);
  return roundf(val * multiplier) / multiplier;
}

void resetEstimations() {
  vx = vy = 0.0;
  px = py = 0.0;
  theta = 0.0;
  for (int i = 0; i < ACCEL_WINDOW; i++) ax_history[i] = ay_history[i] = 0.0;
  for (int i = 0; i < GYRO_WINDOW; i++) wz_history[i] = 0.0;
  for (int i = 0; i < VEL_WINDOW; i++) vx_history[i] = vy_history[i] = 0.0;
  accel_idx = gyro_idx = vel_idx = 0;
}

void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin(11, 17);
  Wire.setClock(50000);

  mpu.begin();
  Serial.println("STATUS,WAITING");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("C,1")) {
      Serial.println("STATUS,CALIBRANDO");
      mpu.calcOffsets();
      imu_calibrated = true;
      last_time = millis();
      resetEstimations();
      state = MEASURING_NOISE;
      noise_start_time = millis();
      max_abs_ax = max_abs_ay = max_abs_wz = 0;
      max_abs_vx = max_abs_vy = 0;
      sum_ax = sum_ay = sum_wz = 0;
      noise_sample_count = 0;
      Serial.println("STATUS,MEASURING_NOISE");
    }
  }

  if (state == WAITING) return;

  mpu.update();
  unsigned long now = millis();
  float dt = (now - last_time) / 1000.0;
  last_time = now;
  if (dt <= 0 || dt > 0.1) return;

  // --- Lectura cruda de sensores ---
  // Aceleración (intercambio X/Y, conversión a m/s² y redondeo a 0.01)
  float ay_raw = roundTo(mpu.getAccX() * 9.80665, 2);
  float ax_raw = roundTo(mpu.getAccY() * 9.80665, 2);
  // Velocidad angular (redondeo a 0.001 rad/s)
  float wz_raw = roundTo(mpu.getGyroZ() * PI / 180.0, 3);

  if (state == MEASURING_NOISE) {
    // Acumular estadísticas de ruido
    max_abs_ax = max(max_abs_ax, fabs(ax_raw));
    max_abs_ay = max(max_abs_ay, fabs(ay_raw));
    max_abs_wz = max(max_abs_wz, fabs(wz_raw));
    sum_ax += ax_raw; sum_ay += ay_raw; sum_wz += wz_raw;

    // Integrar velocidad (todavía sin corregir offset) para medir deriva
    vx += ax_raw * dt;
    vy += ay_raw * dt;
    max_abs_vx = max(max_abs_vx, fabs(vx));
    max_abs_vy = max(max_abs_vy, fabs(vy));
    noise_sample_count++;

    if (now - noise_start_time >= NOISE_DURATION) {
      // Calcular umbrales
      threshold_ax = roundTo(max_abs_ax * 1.2, 2);
      threshold_ay = roundTo(max_abs_ay * 1.2, 2);
      threshold_wz = roundTo(max_abs_wz * 1.2, 3);
      // Umbral de velocidad: margen adicional por la integración de deriva
      threshold_vx = roundTo(max_abs_vx * 1.5, 2);
      threshold_vy = roundTo(max_abs_vy * 1.5, 2);

      // Referencias de offset
      ref_ax = roundTo(sum_ax / noise_sample_count, 2);
      ref_ay = roundTo(sum_ay / noise_sample_count, 2);
      ref_wz = roundTo(sum_wz / noise_sample_count, 3);

      // Reiniciar velocidades acumuladas durante el ruido
      vx = 0; vy = 0;
      for (int i = 0; i < VEL_WINDOW; i++) vx_history[i] = vy_history[i] = 0;

      state = RUNNING;
      Serial.println("STATUS,OK");
      // Cabecera: ax, ay, vx, vy, wz, px, py, theta
      Serial.println("ax,ay,vx,vy,wz,px,py,theta");
    }
    return;
  }

  // --- Corrección de offset (valores ya redondeados) ---
  float ax_corrected = ax_raw - ref_ax;
  float ay_corrected = ay_raw - ref_ay;
  float wz_corrected = wz_raw - ref_wz;

  // --- Filtro de umbral en aceleración y giro ---
  float ax_filt = (fabs(ax_corrected) <= threshold_ax) ? 0.0 : roundTo(ax_corrected, 2);
  float ay_filt = (fabs(ay_corrected) <= threshold_ay) ? 0.0 : roundTo(ay_corrected, 2);
  float wz_filt = (fabs(wz_corrected) <= threshold_wz) ? 0.0 : roundTo(wz_corrected, 3);

  // --- Integración de velocidad lineal (redondeada tras cada paso) ---
  vx += ax_filt * dt;
  vy += ay_filt * dt;
  vx = roundTo(vx, 2);
  vy = roundTo(vy, 2);

  // --- Suavizado de velocidad (media móvil) ---
  vx_history[vel_idx] = vx;
  vy_history[vel_idx] = vy;
  vel_idx = (vel_idx + 1) % VEL_WINDOW;

  int n = (vel_idx == 0) ? VEL_WINDOW : vel_idx;
  if (n == 0) n = 1;
  float vx_smooth = 0, vy_smooth = 0;
  for (int i = 0; i < n; i++) { vx_smooth += vx_history[i]; vy_smooth += vy_history[i]; }
  vx_smooth = roundTo(vx_smooth / n, 2);
  vy_smooth = roundTo(vy_smooth / n, 2);

  // --- Filtro de umbral en velocidad (elimina deriva residual) ---
  vx_smooth = (fabs(vx_smooth) <= threshold_vx) ? 0.0 : vx_smooth;
  vy_smooth = (fabs(vy_smooth) <= threshold_vy) ? 0.0 : vy_smooth;

  // --- Integración de orientación (theta) ---
  theta += wz_filt * dt;
  theta = roundTo(theta, 3);
  while (theta > PI) theta -= 2*PI;
  while (theta < -PI) theta += 2*PI;

  // --- Integración de posición (redondeo tras cada paso) ---
  float dx = (vx_smooth * cos(theta) - vy_smooth * sin(theta)) * dt;
  float dy = (vx_smooth * sin(theta) + vy_smooth * cos(theta)) * dt;
  px += dx;  py += dy;
  px = roundTo(px, 2);
  py = roundTo(py, 2);

  // --- Salida con formatos que ya limitan los decimales mostrados ---
  Serial.printf("%.2f,%.2f,%.2f,%.2f,%.3f,%.2f,%.2f,%.3f\n",
                ax_filt, ay_filt,      // m/s² (2 decimales)
                vx_smooth, vy_smooth,  // m/s (2 decimales)
                wz_filt,               // rad/s (3 decimales)
                px, py, theta);        // m (2 decimales), rad (3 decimales)
}