#include <Arduino_LSM6DSOX.h>
#include <ArduinoBLE.h>
#include <WiFiNINA.h>
#include <Servo.h>
#include <ArduinoBLE.h>
#include "robotMotion.h"
#include "gyro.h"

#define IMU_FAIL_LIMIT 100

JointData pose1 = {
  90,90,90,90,90,90
};

JointData pose2 = {
  90,45,135,90,135,135
};

JointData pose3 = {
  90,45,135,90,135,135
};

robotMotion robot;
gyro imu;

gyroData IMUData;
JointData jData;

// For the LED Blinking thing
int LEDFlag = 0;

float lDt;
unsigned long current_time, prev_time;

// Keep a track of how many times the IMU failed.
int failCount = 0;

int yInput;

// Set up the comunications, sensors and servos before starting the main loop of the program.
void setup()
{
  // Setup the pins.
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  // Set up the serial
  Serial.begin(9600);
  if (!Serial)
  {
    Serial.println("Starting Serial failed!");
  }

  imu.reset();
  robot.reset();

  yInput = 125.0;
  invKinematics(0.0, 125, 0, jData);
  delay(500);
}

void loop()
{

  // while (Serial.available() == 0) {
  // }

  // yInput = Serial.parseInt();

  // if (yInput > 70 && yInput < 300) {
  //   invKinematics(0.0, yInput, 0, jData);
  // }

  prev_time = current_time;
  current_time = micros();
  lDt = (current_time - prev_time) / 1000000.0;

  // Get the data from the IMU.
  // Fail the code if it fails for a large amount of time.
  if (!imu.getData(IMUData))
  {
    delay(10);
    if ((++failCount) > IMU_FAIL_LIMIT)
    {
      Serial.println("Failed to access IMU for 1 sec. Reboot to continue.");
      while (1)
        ;
    }
  }
  else
  {
    failCount = 0;
  }

  // Change the color of the led over time to show the loop is still running.
  cycleLED();
  //debugPrint();

  robot.moveTo(pose1);
  delay(1000);  

  // Loop at max 100Hz
  loopRate(100);
}

// Set a constant update rate regardless of the processes that are running.
void loopRate(int freq)
{
  float invFreq = 1.0 / freq * 1000000.0;
  unsigned long checker = micros();

  // Sit in loop until appropriate time has passed
  while (invFreq > (checker - current_time))
  {
    checker = micros();
  }
}

void debugPrint(void)
{
  Serial.println("Accelerometer data: ");
  Serial.print(IMUData.Ax);
  Serial.print('\t');
  Serial.print(IMUData.Ay);
  Serial.print('\t');
  Serial.println(IMUData.Az);
  Serial.println();

  Serial.println("Gyroscope data: ");
  Serial.print(IMUData.Gx);
  Serial.print('\t');
  Serial.print(IMUData.Gy);
  Serial.print('\t');
  Serial.println(IMUData.Gz);
  Serial.println();

  Serial.println("Temperature: ");
  Serial.print(IMUData.temperature_deg);
  Serial.println();
}

// Cycle the LED values
void cycleLED()
{
  if (LEDFlag == 0)
  {
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, LOW);
  }
  if (LEDFlag == 33)
  {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, LOW);
  }
  if (LEDFlag == 66)
  {
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, HIGH);
  }
  // Serial.println(LEDFlag);
  LEDFlag = (LEDFlag + 1) % 100;
}