#include <Servo.h>

int Servo_Pin = 2;
Servo servo_hip;

void rotateServo(Servo servo, int angle) {
  servo.write(angle);
}

void setup() {
  servo_hip.attach(Servo_Pin);
}
void loop() {
    rotateServo(servo_hip, 0);
    delay(1000);
    rotateServo(servo_hip, 90);
    delay(3000);
    rotateServo(servo_hip, 180);
    delay(1000);
  // for (int pos = 0; pos <= 90; pos += 15) { // goes from 0 degrees to 180 degrees
  //   rotateServo(servo_hip, pos);              // tell servo to go to position in variable 'pos'
  //   delay(1500);                       // waits 1500ms for the servo to reach the position
  // }
  // for (int pos = 90; pos >= 0; pos -= 15) { // goes from 180 degrees to 0 degrees
  //   rotateServo(servo_hip, pos);           // tell servo to go to position in variable 'pos'
  //   delay(1500);                       // waits 1500ms for the servo to reach the position
  // }
}