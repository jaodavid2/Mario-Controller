#include "Particle.h"
#include "adxl343.h"
#include <math.h>

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(AUTOMATIC);

SerialLogHandler logHandler;

const int WINDOW_SIZE = 100;
const int STEP_SIZE = 50;
const unsigned long SAMPLING_INTERVAL_MS = 10;

struct Sample {
  float xa, ya, za;
  float xl, yl, zl;
};

Sample buffer[WINDOW_SIZE];

int sampleIndex = 0;
int samplesSincePrediction = 0;
unsigned long lastSampleTime = 0;

ADXL343 accelerometer_arm(Wire, ADXL343_ADDRESS_ARM);
ADXL343 accelerometer_leg(Wire, ADXL343_ADDRESS_LEG);

bool armOk = false;
bool legOk = false;

String currentPrediction = "unknown";

float magnitude(float x, float y, float z) {
  return sqrt(x * x + y * y + z * z);
}

void calculateStats(float values[], float &mean, float &stddev, float &minVal, float &maxVal, float &range, float &energy) {
  float sum = 0.0;
  float sumSq = 0.0;

  minVal = values[0];
  maxVal = values[0];

  for (int i = 0; i < WINDOW_SIZE; i++) {
    sum += values[i];
    sumSq += values[i] * values[i];

    if (values[i] < minVal) minVal = values[i];
    if (values[i] > maxVal) maxVal = values[i];
  }

  mean = sum / WINDOW_SIZE;
  energy = sumSq / WINDOW_SIZE;
  range = maxVal - minVal;

  float variance = 0.0;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    float diff = values[i] - mean;
    variance += diff * diff;
  }

  stddev = sqrt(variance / WINDOW_SIZE);
}

void addFeatures(float features[], int &index, float values[]) {
  float mean, stddev, minVal, maxVal, range, energy;

  calculateStats(values, mean, stddev, minVal, maxVal, range, energy);

  features[index++] = mean;
  features[index++] = stddev;
  features[index++] = minVal;
  features[index++] = maxVal;
  features[index++] = range;
  features[index++] = energy;
}

void extractFeatures(float features[]) {
  float arm_x[WINDOW_SIZE];
  float arm_y[WINDOW_SIZE];
  float arm_z[WINDOW_SIZE];

  float leg_x[WINDOW_SIZE];
  float leg_y[WINDOW_SIZE];
  float leg_z[WINDOW_SIZE];

  float arm_mag[WINDOW_SIZE];
  float leg_mag[WINDOW_SIZE];

  for (int i = 0; i < WINDOW_SIZE; i++) {
    arm_x[i] = buffer[i].xa;
    arm_y[i] = buffer[i].ya;
    arm_z[i] = buffer[i].za;

    leg_x[i] = buffer[i].xl;
    leg_y[i] = buffer[i].yl;
    leg_z[i] = buffer[i].zl;

    arm_mag[i] = magnitude(arm_x[i], arm_y[i], arm_z[i]);
    leg_mag[i] = magnitude(leg_x[i], leg_y[i], leg_z[i]);
  }

  int index = 0;

  addFeatures(features, index, arm_x);
  addFeatures(features, index, arm_y);
  addFeatures(features, index, arm_z);

  addFeatures(features, index, leg_x);
  addFeatures(features, index, leg_y);
  addFeatures(features, index, leg_z);

  addFeatures(features, index, arm_mag);
  addFeatures(features, index, leg_mag);
}

String predictClass(float f[]) {
  if (f[32] <= -0.35) {
    if (f[40] <= 1.70) {
      if (f[43] <= 0.10) {
        if (f[15] <= 0.45) {
          if (f[11] <= 0.14) {
            return "run";
          } else {
            return "idle";
          }
        } else {
          return "jump";
        }
      } else {
        if (f[16] <= 0.90) {
          if (f[38] <= 0.31) {
            return "jump";
          } else {
            return "walk";
          }
        } else {
          if (f[21] <= 1.94) {
            return "jump";
          } else {
            return "walk";
          }
        }
      }
    } else {
      if (f[6] <= -0.64) {
        if (f[34] <= 6.01) {
          if (f[42] <= 1.31) {
            return "jump";
          } else {
            return "run";
          }
        } else {
          if (f[42] <= 1.63) {
            return "jump";
          } else {
            return "run";
          }
        }
      } else {
        if (f[18] <= 1.07) {
          return "jump";
        } else {
          if (f[29] <= 0.70) {
            return "jump";
          } else {
            return "run";
          }
        }
      }
    }
  } else {
    if (f[12] <= 0.64) {
      if (f[29] <= 0.01) {
        return "jump";
      } else {
        return "idle";
      }
    } else {
      if (f[26] <= 0.53) {
        return "jump";
      } else {
        return "idle";
      }
    }
  }
}

void shiftBufferLeft() {
  for (int i = STEP_SIZE; i < WINDOW_SIZE; i++) {
    buffer[i - STEP_SIZE] = buffer[i];
  }

  sampleIndex = WINDOW_SIZE - STEP_SIZE;
}

void setup() {
  Serial.begin(9600);
  waitFor(Serial.isConnected, 10000);

  delay(1000);

  armOk = accelerometer_arm.begin();
  legOk = accelerometer_leg.begin();

  Particle.variable("prediction", currentPrediction);

  Serial.println("Mario Controller TinyML started");
  Serial.printlnf("ARM sensor: %s", armOk ? "OK" : "FAIL");
  Serial.printlnf("LEG sensor: %s", legOk ? "OK" : "FAIL");
  Serial.println("Sampling started...");
}

void loop() {
  if (millis() - lastSampleTime < SAMPLING_INTERVAL_MS) {
    return;
  }

  lastSampleTime = millis();

  float xa = 0, ya = 0, za = 0;
  float xl = 0, yl = 0, zl = 0;

  if (armOk) {
    accelerometer_arm.readAccelerationG(&xa, &ya, &za);
  }

  if (legOk) {
    accelerometer_leg.readAccelerationG(&xl, &yl, &zl);
  }

  buffer[sampleIndex].xa = xa;
  buffer[sampleIndex].ya = ya;
  buffer[sampleIndex].za = za;

  buffer[sampleIndex].xl = xl;
  buffer[sampleIndex].yl = yl;
  buffer[sampleIndex].zl = zl;

  sampleIndex++;

  if (sampleIndex >= WINDOW_SIZE) {
    float features[48];

    extractFeatures(features);

    currentPrediction = predictClass(features);

    Serial.print("Prediction: ");
    Serial.println(currentPrediction);

    shiftBufferLeft();
  }
}