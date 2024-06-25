#include "robotMotion.h"

robotMotion::robotMotion()
{
    // Right servos
    servo_1_r.attach(SERVO_1_R);
    servo_2_r.attach(SERVO_2_R);
    servo_3_r.attach(SERVO_3_R);

    // Left servos
    servo_1_l.attach(SERVO_1_L);
    servo_2_l.attach(SERVO_2_L);
    servo_3_l.attach(SERVO_3_L);
}

bool robotMotion::verifyAngles(const JointData data)
{
    if (data.lj1 < 45 || data.lj1 > 135)
    {
        return false;
    }
    if (data.rj1 < 45 || data.rj1 > 135)
    {
        return false;
    }

    if (data.lj2 < 20 || data.lj2 > 150)
    {
        return false;
    }
    if (data.rj2 < 20 || data.rj2 > 150)
    {
        return false;
    }

    if (data.lj3 < 45 || data.lj3 > 135)
    {
        return false;
    }
    if (data.rj3 < 45 || data.rj3 > 135)
    {
        return false;
    }

    return true;
}

void robotMotion::reset()
{
    servo_1_r.write(90);
    servo_2_r.write(90);
    servo_3_r.write(90);

    servo_1_l.write(90);
    servo_2_l.write(90);
    servo_3_l.write(90);
}

void robotMotion::moveTo(const JointData data)
{

    if (this->curState == data)
    {
        return;
    }

    if (!this->verifyAngles(data))
    {
        return;
    }

    // Right leg    
    servo_1_r.write(data.rj1);
    servo_2_r.write(data.rj2);
    servo_3_r.write(data.rj3 - data.rj2);

    // Left Leg
    servo_1_l.write(data.lj1);
    servo_2_l.write(data.lj2);
    servo_3_l.write(data.lj3);

    this->curState = data;
}


void invKinematics(float x, float y, float z, JointData &data) {
  float gamma = 0.0f;
  float beta = 0.0f;
  float alpha = 3.14/2;

  float legLength = sqrt(x*x + y*y);
  float tempBeta = acos((L2*L2 + legLength*legLength - L3*L3)/(2*L2*legLength));
  if (y < 0.01 && y > -0.1) {
    y = 0.01;
  }
  float theta = atan((x)/(y));


  beta = theta+tempBeta;
  gamma = theta-tempBeta;
  data.rj1 = alpha * 57.3;
  data.rj2 = 90 - (beta * 57.3);
  data.rj3 = -(gamma * 57.3);

  data.lj1 = alpha * 57.3;
  data.lj2 = beta * 57.3;
  data.lj3 = gamma * 57.3;
}

