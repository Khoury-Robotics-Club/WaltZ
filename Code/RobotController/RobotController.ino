#include <Arduino_LSM6DSOX.h>
#include <ArduinoBLE.h>
#include <WiFiNINA.h>


#define IMU_FAIL_LIMIT 100

// For the LED Blinking thing
int LEDFlag = 0;

// Value between 0 and 1 for the madwig filter. 
// 1 -> noisy data, 0 -> low response 
float beta = 0.05;


float Ax, Ay, Az;
float Gx, Gy, Gz;

float AxPrev, AyPrev, AzPrev;
float GxPrev, GyPrev, GzPrev;

int temperature_deg = 0;

float AxError, AyError, AzError;
float GxError, GyError, GzError;


float q0 = 1.0f; //initialize quaternion for madgwick filter
float q1 = 0.0f;
float q2 = 0.0f;
float q3 = 0.0f;


float roll_IMU, pitch_IMU, yaw_IMU;
float roll_IMU_prev, pitch_IMU_prev, yaw_IMU_prev;

float lDt;
unsigned long current_time, prev_time;

// Keep a track of how many times the IMU failed.
int failCount = 0;

// Set up the comunications, sensors and servos before starting the main loop of the program.
void setup() {
  // Setup the pins.
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  
  // Set up the serial
  Serial.begin(9600);
  if(!Serial) {
    Serial.println("Starting Serial failed!");
  }

  // Set up the bluetooth connection.
  if (!BLE.begin()) {
    Serial.println("Starting Bluetooth® Low Energy failed!");
  }

  // Max sample rate is 104Hz
  // Cap the refresh rate of the bot to 100Hz
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Get the error of the IMU. Helps with drift.
  calculateIMUError();
}

void loop() {
  // To calculate the physics we calculate the delta time.
  prev_time = current_time;      
  current_time = micros();      
  lDt = (current_time - prev_time)/1000000.0; 

  // Get the data from the IMU.
  // Fail the code if it fails for a large amount of time.
  if(!getIMUData()) {
    delay(10);
    if ((++failCount) > IMU_FAIL_LIMIT) {
      Serial.println("Failed to access IMU for 1 sec. Reboot to continue.");
      while(1);
    }
  } else {
    failCount = 0;
  }

  // Use the madwig filter to get the values.
  Madgwick(Gx,Gy,Gz,Ax,Ay,Az,lDt);

  // Set the values for the previous variables.
  setPrevValues();

  // Change the color of the led over time to show the loop is still running.
  cycleLED();
  debugPrint();
  
  // Loop at max 500Hz
  loopRate(100);
}


// Get the data from the IMU.
bool getIMUData () {
  if (IMU.accelerationAvailable()) {
    // In terms of Gs
    IMU.readAcceleration(Ax, Ay, Az);
  } else {
    return false;
  }

  if (IMU.gyroscopeAvailable()) {
    // In degree/sec
    IMU.readGyroscope(Gx, Gy, Gz);
  } else {
    return false;
  }

  if (IMU.temperatureAvailable()) {
    IMU.readTemperature(temperature_deg);
  } else {
    return false;
  }

  Ax = Ax-AxError;
  Ay = Ay-AyError;

  Gx = Gx-GxError;
  Gy = Gy-GyError;
  Gz = Gz-GzError;

  return true;
}

//Madwig filter for the IMU.
// Taken from dRehmFlight.
void Madgwick(float gx, float gy, float gz, float ax, float ay, float az, float invSampleFreq) {
  float recipNorm;
  float s0, s1, s2, s3;
  float qDot1, qDot2, qDot3, qDot4;
  float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2 ,_8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

  //Convert gyroscope degrees/sec to radians/sec
  gx *= 0.0174533f;
  gy *= 0.0174533f;
  gz *= 0.0174533f;

  //Rate of change of quaternion from gyroscope
  qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
  qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
  qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
  qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

  //Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
  if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
    //Normalise accelerometer measurement
    recipNorm = invSqrt(ax * ax + ay * ay + az * az);
    ax *= recipNorm;
    ay *= recipNorm;
    az *= recipNorm;

    //Auxiliary variables to avoid repeated arithmetic
    _2q0 = 2.0f * q0;
    _2q1 = 2.0f * q1;
    _2q2 = 2.0f * q2;
    _2q3 = 2.0f * q3;
    _4q0 = 4.0f * q0;
    _4q1 = 4.0f * q1;
    _4q2 = 4.0f * q2;
    _8q1 = 8.0f * q1;
    _8q2 = 8.0f * q2;
    q0q0 = q0 * q0;
    q1q1 = q1 * q1;
    q2q2 = q2 * q2;
    q3q3 = q3 * q3;

    //Gradient decent algorithm corrective step
    s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
    s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
    s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
    s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;
    recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3); //normalise step magnitude
    s0 *= recipNorm;
    s1 *= recipNorm;
    s2 *= recipNorm;
    s3 *= recipNorm;

    //Apply feedback step
    qDot1 -= beta * s0;
    qDot2 -= beta * s1;
    qDot3 -= beta * s2;
    qDot4 -= beta * s3;
  }

  //Integrate rate of change of quaternion to yield quaternion
  q0 += qDot1 * invSampleFreq;
  q1 += qDot2 * invSampleFreq;
  q2 += qDot3 * invSampleFreq;
  q3 += qDot4 * invSampleFreq;

  //Normalise quaternion
  recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  q0 *= recipNorm;
  q1 *= recipNorm;
  q2 *= recipNorm;
  q3 *= recipNorm;

  //compute angles
  roll_IMU = atan2(q0*q1 + q2*q3, 0.5f - q1*q1 - q2*q2)*57.29577951;
  pitch_IMU = asin(-2.0f * (q1*q3 - q0*q2))*57.29577951;
  yaw_IMU = atan2(q1*q2 + q0*q3, 0.5f - q2*q2 - q3*q3)*57.29577951;
}

void setPrevValues() {
  roll_IMU_prev = roll_IMU;
  pitch_IMU_prev = pitch_IMU;
  yaw_IMU_prev = yaw_IMU;
}


// Calculate an offset for the IMU.
void calculateIMUError() {
  for (int i=0;i<1000;i++) {
    if (getIMUData()) {
      AxError += Ax;
      AyError += Ay;
      AzError += Az;
      
      GxError += Gx;
      GyError += Gy;
      GzError += Gz;
    } else {
      Serial.print("Error in calculating IMU error.");
      break;
    } 
  }
}


// Set a constant update rate regardless of the processes that are running.
void loopRate(int freq) {
  float invFreq = 1.0/freq*1000000.0;
  unsigned long checker = micros();
  
  //Sit in loop until appropriate time has passed
  while (invFreq > (checker - current_time)) {
    checker = micros();
  }
}

void debugPrint(void) {
  // Serial.println("Accelerometer data: ");
  // Serial.print(Ax);
  // Serial.print('\t');
  // Serial.print(Ay);
  // Serial.print('\t');
  // Serial.println(Az);
  // Serial.println();
  
  // Serial.println("Gyroscope data: ");
  // Serial.print(Gx);
  // Serial.print('\t');
  // Serial.print(Gy);
  // Serial.print('\t');
  // Serial.println(Gz);
  // Serial.println();

  // Serial.println("Temperature: ");
  // Serial.print(temperature_deg);
  // Serial.println();

  Serial.println("Madwig output: ");
  Serial.print(roll_IMU);
  Serial.print('\t');
  Serial.print(pitch_IMU);
  Serial.print('\t');
  Serial.print(yaw_IMU);
  Serial.println();

}

// Inverse square root of the number.
float invSqrt(float in) {
  return 1/sqrt(in);
}

// Cycle the LED values
void cycleLED() {
  if (LEDFlag == 0) {
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, LOW);
  }
  if (LEDFlag == 333) {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, LOW);
  }
  if (LEDFlag == 666) {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, HIGH);
  }
  //Serial.println(LEDFlag);
  LEDFlag = (LEDFlag+1)%1000;
}