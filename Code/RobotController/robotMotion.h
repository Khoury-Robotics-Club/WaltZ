/**
 * This class represents the bot.
 * It allows us to move the bot to known states and control the bot to any arbitrary positions.
 */

#include <Servo.h>

#define SERVO_1_R 2
#define SERVO_2_R 3
#define SERVO_3_R 4

#define SERVO_1_L 5
#define SERVO_2_L 6
#define SERVO_3_L 7

// In MM
#define L1 15
#define L2 150
#define L3 150

struct JointData
{
    int rj1, rj2, rj3;
    int lj1, lj2, lj3;

    bool operator==(const JointData &other) const
    {
        return rj1 == other.rj1 &&
               rj2 == other.rj2 &&
               rj3 == other.rj3 &&
               lj1 == other.lj1 &&
               lj2 == other.lj2 &&
               lj3 == other.lj3;
    }
};

class robotMotion
{
private:
    Servo servo_1_r;
    Servo servo_2_r;
    Servo servo_3_r;

    Servo servo_1_l;
    Servo servo_2_l;
    Servo servo_3_l;

    JointData curState;

public:
    bool verifyAngles(const JointData data);
    robotMotion();
    void reset();
    void moveTo(const JointData data);
};