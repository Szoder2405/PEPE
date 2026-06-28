#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_light.h>

// ==================== PINES MOTORES DC ====================
#define LEFT_IN1    47
#define LEFT_IN2    48
#define LEFT_PWM    42
#define RIGHT_IN1   39
#define RIGHT_IN2   38
#define RIGHT_PWM   37

// ==================== PINES ENCODERS (HALL) ====================
#define LEFT_ENC_A  36
#define LEFT_ENC_B  35
#define RIGHT_ENC_A 2
#define RIGHT_ENC_B 1

// ==================== CONSTANTES GENERALES ====================
#define BAUD_RATE           115200
#define PWM_MAX             255
#define CONTROL_PERIOD_MS   50     // 20 Hz
#define PULSES_PER_REV       485
#define WHEEL_RADIUS         0.045
#define WHEELBASE            0.17

// Filtro media móvil para encoders
#define ENC_FILTER_WINDOW    10

// ==================== DATOS DE CALIBRACIÓN DEL MOTOR ====================
const float cal_curve[][2] = {
  {0.0,   0},
  {0.180, 130},
  {0.300, 150},
  {0.400, 170},
  {0.470, 200},
  {0.800, 255}
};
const int cal_points = sizeof(cal_curve) / sizeof(cal_curve[0]);

// ==================== ESTADOS DEL SISTEMA ====================
enum State {WAITING, CALIBRATING, RUNNING};
State state = WAITING;

// ==================== OBJETOS Y VARIABLES DEL MPU6050 ====================
MPU6050 mpu(Wire);
bool imu_calibrated = false;
unsigned long last_mpu_time = 0;

#define GYRO_WINDOW 10
float wz_history[GYRO_WINDOW] = {0};
int gyro_idx = 0;
bool gyro_buffer_full = false;

float threshold_wz = 0.5;
float ref_wz = 0;
float wz_imu = 0.0;

unsigned long noise_start_time = 0;
const unsigned long NOISE_DURATION = 3000;
float max_abs_wz = 0;
float sum_wz = 0;
int noise_sample_count = 0;

// ==================== VARIABLES DE ENCODERS ====================
volatile long left_pulses = 0;
volatile long right_pulses = 0;
long prev_left_pulses = 0, prev_right_pulses = 0;
unsigned long last_odom_time = 0;

float v_enc_filtered = 0.0;
float w_enc_filtered = 0.0;

float v_enc_history[ENC_FILTER_WINDOW] = {0};
float w_enc_history[ENC_FILTER_WINDOW] = {0};
int enc_filter_idx = 0;
bool enc_filter_ready = false;

// ==================== FILTROS DE KALMAN ====================
float v_fused = 0.0;
float w_fused = 0.0;
float P_w = 1.0;
const float Q_w = 0.01;
const float R_w_enc = 0.1;
const float R_w_imu = 0.01;

// ==================== CONTROL PID ====================
float v_target = 0.0;
float w_target = 0.0;

float Kp_linear = 100.0, Ki_linear = 500, Kd_linear = 30.0;
float Kp_angular = 15.0, Ki_angular = 50, Kd_angular = 1.0;

float integral_v = 0.0, prev_error_v = 0.0;
float integral_w = 0.0, prev_error_w = 0.0;
const float integral_limit = 100.0;

bool pid_enabled = true;
int manual_pwm_left = 0;
int manual_pwm_right = 0;

// ==================== POSE ESTIMADA ====================
float x_est = 0.0, y_est = 0.0, theta_est = 0.0;

// ==================== FUNCIONES AUXILIARES ====================
inline float roundTo(float val, int decimals) {
  float multiplier = powf(10.0, decimals);
  return roundf(val * multiplier) / multiplier;
}

void resetEstimations() {
  v_enc_filtered = 0.0;
  w_enc_filtered = 0.0;
  v_fused = w_fused = 0.0;
  P_w = 1.0;
  x_est = y_est = theta_est = 0.0;
  for (int i = 0; i < GYRO_WINDOW; i++) wz_history[i] = 0.0;
  for (int i = 0; i < ENC_FILTER_WINDOW; i++) {
    v_enc_history[i] = w_enc_history[i] = 0.0;
  }
  gyro_idx = 0;
  gyro_buffer_full = false;
  enc_filter_idx = 0;
  enc_filter_ready = false;
}

int pwmForSpeed(float v_target) {
  if (v_target <= 0) return 0;
  if (v_target >= cal_curve[cal_points-1][0]) return (int)cal_curve[cal_points-1][1];
  for (int i = 0; i < cal_points-1; i++) {
    float v1 = cal_curve[i][0];
    float pwm1 = cal_curve[i][1];
    float v2 = cal_curve[i+1][0];
    float pwm2 = cal_curve[i+1][1];
    if (v_target >= v1 && v_target <= v2) {
      float t = (v_target - v1) / (v2 - v1);
      return (int)(pwm1 + t * (pwm2 - pwm1));
    }
  }
  return 0;
}

// ==================== INTERRUPCIONES ENCODERS ====================
void IRAM_ATTR leftEncoderISR() {
  if (digitalRead(LEFT_ENC_B) == HIGH) left_pulses++;
  else left_pulses--;
}
void IRAM_ATTR rightEncoderISR() {
  if (digitalRead(RIGHT_ENC_B) == HIGH) right_pulses++;
  else right_pulses--;
}

// ==================== CONTROL MOTORES ====================
void setMotorLeft(int speed, bool forward) {
  ledcWrite(LEFT_PWM, speed);
  digitalWrite(LEFT_IN1, forward ? HIGH : LOW);
  digitalWrite(LEFT_IN2, forward ? LOW : HIGH);
}
void setMotorRight(int speed, bool forward) {
  ledcWrite(RIGHT_PWM, speed);
  digitalWrite(RIGHT_IN1, forward ? HIGH : LOW);
  digitalWrite(RIGHT_IN2, forward ? LOW : HIGH);
}
void stopMotors() {
  ledcWrite(LEFT_PWM, 0); ledcWrite(RIGHT_PWM, 0);
  digitalWrite(LEFT_IN1, LOW); digitalWrite(LEFT_IN2, LOW);
  digitalWrite(RIGHT_IN1, LOW); digitalWrite(RIGHT_IN2, LOW);
}
void setMotorLeftRaw(int pwm_signed) {
  if (pwm_signed >= 0) setMotorLeft(pwm_signed, true);
  else setMotorLeft(-pwm_signed, false);
}
void setMotorRightRaw(int pwm_signed) {
  if (pwm_signed >= 0) setMotorRight(pwm_signed, true);
  else setMotorRight(-pwm_signed, false);
}

// ==================== COMANDOS SERIE ====================
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("OFF")) { pid_enabled = false; Serial.println("PID OFF"); return; }
  if (cmd.startsWith("ON")) { pid_enabled = true; integral_v=integral_w=0; prev_error_v=prev_error_w=0; Serial.println("PID ON"); return; }

  if (cmd.charAt(0) == 'F' || cmd.charAt(0) == 'B' || cmd.charAt(0) == 'L' || cmd.charAt(0) == 'R' || cmd.charAt(0) == 'S') {
    char action = cmd.charAt(0);
    float value = cmd.substring(2).toFloat();
    value = constrain(value, 0.0, 9.8);
    switch (action) {
      case 'F': v_target = value; w_target = 0.0; break;
      case 'B': v_target = -value; w_target = 0.0; break;
      case 'L': v_target = 0.0; w_target = value; break;
      case 'R': v_target = 0.0; w_target = -value; break;
      case 'S': v_target = w_target = 0.0; if (!pid_enabled) { manual_pwm_left = manual_pwm_right = 0; } break;
    }
    Serial.printf("Consigna: v=%.2f w=%.2f\n", v_target, w_target);
    return;
  }

  if (cmd.startsWith("PWM,")) {
    int c1 = cmd.indexOf(','), c2 = cmd.indexOf(',', c1+1);
    if (c2 != -1) {
      int l = cmd.substring(c1+1, c2).toInt(); l = constrain(l, -255, 255);
      int r = cmd.substring(c2+1).toInt(); r = constrain(r, -255, 255);
      manual_pwm_left = l; manual_pwm_right = r;
      Serial.printf("PWM L=%d R=%d\n", l, r);
    }
    return;
  }
  if (cmd.startsWith("PWM_L,")) { int v = cmd.substring(6).toInt(); manual_pwm_left = constrain(v, -255,255); Serial.printf("PWM_L=%d\n", v); return; }
  if (cmd.startsWith("PWM_R,")) { int v = cmd.substring(6).toInt(); manual_pwm_right = constrain(v, -255,255); Serial.printf("PWM_R=%d\n", v); return; }

  if (cmd.startsWith("KP,"))  { Kp_linear = cmd.substring(3).toFloat(); Serial.printf("Kp_l=%.2f\n", Kp_linear); }
  else if (cmd.startsWith("KI,"))  { Ki_linear = cmd.substring(3).toFloat(); Serial.printf("Ki_l=%.2f\n", Ki_linear); }
  else if (cmd.startsWith("KD,"))  { Kd_linear = cmd.substring(3).toFloat(); Serial.printf("Kd_l=%.2f\n", Kd_linear); }
  else if (cmd.startsWith("KPW,")) { Kp_angular = cmd.substring(4).toFloat(); Serial.printf("Kp_a=%.2f\n", Kp_angular); }
  else if (cmd.startsWith("KIW,")) { Ki_angular = cmd.substring(4).toFloat(); Serial.printf("Ki_a=%.2f\n", Ki_angular); }
  else if (cmd.startsWith("KDW,")) { Kd_angular = cmd.substring(4).toFloat(); Serial.printf("Kd_a=%.2f\n", Kd_angular); }
  else if (cmd.startsWith("RI,")) { integral_v=integral_w=0; prev_error_v=prev_error_w=0; Serial.println("Integrales reseteadas"); }
  else if (cmd == "PRINT") {
    Serial.printf("PID: %s\n", pid_enabled?"ON":"OFF");
    Serial.printf("Kp_l=%.2f Ki_l=%.2f Kd_l=%.2f\n", Kp_linear, Ki_linear, Kd_linear);
    Serial.printf("Kp_a=%.2f Ki_a=%.2f Kd_a=%.2f\n", Kp_angular, Ki_angular, Kd_angular);
    if (!pid_enabled) Serial.printf("PWM manual: L=%d R=%d\n", manual_pwm_left, manual_pwm_right);
  }
  else if (cmd.startsWith("C,1")) {
    Serial.println("STATUS,CALIBRANDO");
    mpu.calcOffsets();
    imu_calibrated = true;
    last_mpu_time = millis();
    resetEstimations();
    state = CALIBRATING;
    noise_start_time = millis();
    max_abs_wz = 0;
    sum_wz = 0;
    noise_sample_count = 0;
    Serial.println("STATUS,MEASURING_NOISE");
  }
  else if (cmd.startsWith("X,1")) {
    ESP.restart();
  }
  else if (cmd.startsWith("Z,0")) {
    x_est = y_est = theta_est = 0;
    Serial.println("Pose reiniciada");
  }
  else {
    Serial.println("Cmd: F/B/L/R/S,val | ON/OFF | PWM... | KP/KI/KD/KPW/KIW/KDW,val | RI,0 | C,1 | X,1 | Z,0 | PRINT");
  }
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin(11, 17);
  Wire.setClock(50000);

  mpu.begin();
  Serial.println("STATUS,WAITING");
  imu_calibrated = false;

  pinMode(LEFT_IN1, OUTPUT); pinMode(LEFT_IN2, OUTPUT); pinMode(LEFT_PWM, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT); pinMode(RIGHT_IN2, OUTPUT); pinMode(RIGHT_PWM, OUTPUT);
  stopMotors();
  ledcAttach(LEFT_PWM, 5000, 8);
  ledcAttach(RIGHT_PWM, 5000, 8);

  pinMode(LEFT_ENC_A, INPUT_PULLUP); pinMode(LEFT_ENC_B, INPUT_PULLUP);
  pinMode(RIGHT_ENC_A, INPUT_PULLUP); pinMode(RIGHT_ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_A), leftEncoderISR, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_A), rightEncoderISR, RISING);

  last_odom_time = millis();
  prev_left_pulses = left_pulses;
  prev_right_pulses = right_pulses;
}

// ==================== LOOP ====================
void loop() {
  // Comandos serie
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }

  unsigned long now = millis();

  // Control periódico cada CONTROL_PERIOD_MS
  if (now - last_odom_time >= CONTROL_PERIOD_MS) {
    float dt = (now - last_odom_time) / 1000.0;
    last_odom_time = now;

    // ----- Lectura de encoders -----
    long dL, dR;
    noInterrupts();
    dL = left_pulses - prev_left_pulses;
    dR = right_pulses - prev_right_pulses;
    prev_left_pulses  = left_pulses;
    prev_right_pulses = right_pulses;
    interrupts();

    float wL = (dL / (float)PULSES_PER_REV) * 2.0 * PI / dt;
    float wR = (dR / (float)PULSES_PER_REV) * 2.0 * PI / dt;
    float v_left_actual = WHEEL_RADIUS * wL;
    float v_right_actual = WHEEL_RADIUS * wR;

    float v_enc_raw = (v_left_actual + v_right_actual) / 2.0;
    float w_enc_raw = (v_right_actual - v_left_actual) / WHEELBASE;

    // Filtro media móvil
    v_enc_history[enc_filter_idx] = v_enc_raw;
    w_enc_history[enc_filter_idx] = w_enc_raw;
    enc_filter_idx = (enc_filter_idx + 1) % ENC_FILTER_WINDOW;
    if (enc_filter_idx == 0) enc_filter_ready = true;

    if (enc_filter_ready) {
      float sum_v = 0, sum_w = 0;
      for (int i = 0; i < ENC_FILTER_WINDOW; i++) { sum_v += v_enc_history[i]; sum_w += w_enc_history[i]; }
      v_enc_filtered = sum_v / ENC_FILTER_WINDOW;
      w_enc_filtered = sum_w / ENC_FILTER_WINDOW;
    } else {
      int n = enc_filter_idx; if (n == 0) n = 1;
      float sum_v = 0, sum_w = 0;
      for (int i = 0; i < enc_filter_idx; i++) { sum_v += v_enc_history[i]; sum_w += w_enc_history[i]; }
      v_enc_filtered = sum_v / n;
      w_enc_filtered = sum_w / n;
    }

    // ----- Lectura del giroscopio (MPU6050) -----
    if (imu_calibrated && (state == CALIBRATING || state == RUNNING)) {
      mpu.update();
      float dt_mpu = (now - last_mpu_time) / 1000.0;
      last_mpu_time = now;
      if (dt_mpu > 0 && dt_mpu < 0.1) {
        float wz_raw = roundTo(mpu.getGyroZ() * PI / 180.0, 3);

        if (state == CALIBRATING) {
          max_abs_wz = max(max_abs_wz, fabs(wz_raw));
          sum_wz += wz_raw;
          noise_sample_count++;

          if (now - noise_start_time >= NOISE_DURATION) {
            threshold_wz = roundTo(max_abs_wz * 1.2, 3);
            ref_wz = roundTo(sum_wz / noise_sample_count, 3);
            state = RUNNING;
            Serial.println("STATUS,OK");
          }
          return;
        }

        float wz_corr = wz_raw - ref_wz;
        wz_imu = (fabs(wz_corr) <= threshold_wz) ? 0.0 : roundTo(wz_corr, 3);

        wz_history[gyro_idx] = wz_imu;
        gyro_idx = (gyro_idx + 1) % GYRO_WINDOW;
        if (gyro_idx == 0) gyro_buffer_full = true;
      }
    }

    // ----- Fusión Kalman -----
    if (state != RUNNING) {
      v_fused = v_enc_filtered;
      w_fused = w_enc_filtered;
    } else {
      v_fused = v_enc_filtered;
      P_w += Q_w;
      float K_w_enc = P_w / (P_w + R_w_enc);
      w_fused += K_w_enc * (w_enc_filtered - w_fused);
      P_w = (1 - K_w_enc) * P_w;
      float K_w_imu = P_w / (P_w + R_w_imu);
      w_fused += K_w_imu * (wz_imu - w_fused);
      P_w = (1 - K_w_imu) * P_w;
    }

    // ----- Control PID con feedforward -----
    if (pid_enabled) {
      float error_v = v_target - v_fused;
      if (fabs(v_target) < 0.001 && fabs(error_v) < 0.01) integral_v = 0;
      integral_v += error_v * dt;
      integral_v = constrain(integral_v, -integral_limit, integral_limit);
      float derivative_v = (error_v - prev_error_v) / dt;
      float u_v_corr = Kp_linear * error_v + Ki_linear * integral_v + Kd_linear * derivative_v;
      prev_error_v = error_v;

      float error_w = w_target - w_fused;
      if (fabs(w_target) < 0.001 && fabs(error_w) < 0.01) integral_w = 0;
      integral_w += error_w * dt;
      integral_w = constrain(integral_w, -integral_limit, integral_limit);
      float derivative_w = (error_w - prev_error_w) / dt;
      float u_w_corr = Kp_angular * error_w + Ki_angular * integral_w + Kd_angular * derivative_w;
      prev_error_w = error_w;

      float v_ff_ang = fabs(w_target) * (WHEELBASE / 2.0);
      int pwm_ff_ang = pwmForSpeed(v_ff_ang);
      int pwm_left_ang  = (w_target >= 0) ? -pwm_ff_ang : pwm_ff_ang;
      int pwm_right_ang = (w_target >= 0) ? pwm_ff_ang : -pwm_ff_ang;
      pwm_left_ang  -= (int)u_w_corr;
      pwm_right_ang += (int)u_w_corr;

      int pwm_base = (v_target >= 0) ? pwmForSpeed(fabs(v_target)) : -pwmForSpeed(fabs(v_target));

      int pwm_left  = pwm_base + pwm_left_ang  + (int)u_v_corr;
      int pwm_right = pwm_base + pwm_right_ang + (int)u_v_corr;

      if (pwm_left < 0)  setMotorLeft(-pwm_left, false);
      else               setMotorLeft(pwm_left, true);
      if (pwm_right < 0) setMotorRight(-pwm_right, false);
      else               setMotorRight(pwm_right, true);
    } else {
      setMotorLeftRaw(manual_pwm_left);
      setMotorRightRaw(manual_pwm_right);
    }

    // ----- Integración de pose -----
    float v_fused_rounded = roundTo(v_fused, 2);
    float w_fused_rounded = roundTo(w_fused, 3);

    theta_est += w_fused_rounded * dt;
    theta_est = roundTo(theta_est, 3);
    while (theta_est > PI) theta_est -= 2*PI;
    while (theta_est < -PI) theta_est += 2*PI;

    float dx = v_fused_rounded * cos(theta_est) * dt;
    float dy = v_fused_rounded * sin(theta_est) * dt;
    x_est += dx;
    y_est += dy;
    x_est = roundTo(x_est, 2);
    y_est = roundTo(y_est, 2);

    // ----- Envío de datos -----
    Serial.printf("EKF,%.2f,%.2f,%.3f\n", x_est, y_est, theta_est);
    Serial.printf("VEL,%.2f,%.2f,%.3f,%.3f\n", v_target, roundTo(v_fused, 2), w_target, roundTo(w_fused, 3));
  }

  delay(5);
}