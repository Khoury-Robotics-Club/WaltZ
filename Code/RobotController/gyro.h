#include <Arduino_LSM6DSOX.h>

#define dampingFactor 0.95

struct gyroData
{
    float Ax, Ay, Az;
    float Gx, Gy, Gz;

    int temperature_deg;
};

class gyro
{
private:
    float Ax, Ay, Az;
    float Gx, Gy, Gz;

    float AxPrev, AyPrev, AzPrev;
    float GxPrev, GyPrev, GzPrev;

    float AxError, AyError, AzError;
    float GxError, GyError, GzError;

    int temperature_deg;

public:
    gyro();
    void reset();
    bool getData(gyroData &data);
};