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

    if (data.lj2 < 45 || data.lj2 > 135)
    {
        return false;
    }
    if (data.rj2 < 45 || data.rj2 > 135)
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

    servo_1_r.write(data.rj1);
    servo_2_r.write(data.rj2);
    servo_3_r.write(data.rj3);

    servo_1_l.write(data.lj1);
    servo_2_l.write(data.lj2);
    servo_3_l.write(data.lj3);

    this->curState = data;
}