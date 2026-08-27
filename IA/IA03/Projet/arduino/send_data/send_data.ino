#include "Arduino_BMI270_BMM150.h"

const int SAMPLE_WARMUP = 3;

const int SAMPLE_COUNT = 50;
const int SAMPLE_LENGTH = 100;

float x[SAMPLE_COUNT * SAMPLE_LENGTH]{};
float y[SAMPLE_COUNT * SAMPLE_LENGTH]{};
float z[SAMPLE_COUNT * SAMPLE_LENGTH]{};


void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);


  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Serial.print("Accelerometer sample rate = ");
  // Serial.print(IMU.accelerationSampleRate());
  // Serial.println(" Hz");
  // Serial.println();
  // Serial.println("Acceleration in G's");
  // Serial.println("X\tY\tZ");
}

void loop() {
  while(!Serial.available()) {}
  while(Serial.available()) {
    Serial.read();
  }

  for (int i = 0; i < SAMPLE_WARMUP; i++) {
    analogWrite(LEDR, 0);
    analogWrite(LEDG, 255);
    analogWrite(LEDB, 255);
    delay(1000);
    analogWrite(LEDR, 255);
    analogWrite(LEDG, 255);
    analogWrite(LEDB, 255);
    delay(1000);
  }

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    analogWrite(LEDR, 255);
    analogWrite(LEDG, 0);
    analogWrite(LEDB, 255);
    for (int j = 0; j < SAMPLE_LENGTH; j++) {
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x[i * SAMPLE_LENGTH + j], y[i * SAMPLE_LENGTH + j], z[i * SAMPLE_LENGTH + j]);
      }
      delay(10);
    }
    analogWrite(LEDR, 255);
    analogWrite(LEDG, 255);
    analogWrite(LEDB, 255);
    delay(1000);
  }

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    for (int j = 0; j < SAMPLE_LENGTH; j++) {
      Serial.print(x[i * SAMPLE_LENGTH + j]);
      Serial.print('\t');
      Serial.print(y[i * SAMPLE_LENGTH + j]);
      Serial.print('\t');
      Serial.println(z[i * SAMPLE_LENGTH + j]);
    }
  }
}