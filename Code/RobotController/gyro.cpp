#include "gyro.h"

// Init the gyro and set the offsets by taking 1000 values.
gyro::gyro()
{
    // Init all the variables to zero.
    Ax = 0.0f;
    Ay = 0.0f;
    Az = 0.0f;

    Gx = 0.0f;
    Gy = 0.0f;
    Gz = 0.0f;

    AxPrev = 0.0f;
    AyPrev = 0.0f;
    AzPrev = 0.0f;

    GxPrev = 0.0f;
    GyPrev = 0.0f;
    GzPrev = 0.0f;

    AxError = 0.0f;
    AyError = 0.0f;
    AzError = 0.0f;

    GxError = 0.0f;
    GyError = 0.0f;
    GzError = 0.0f;
}

void gyro::reset()
{
    if (!IMU.begin())
    {
        while (1)
            ;
    }

    for (int idx = 0; idx < 1000; ++idx)
    {
        if (IMU.accelerationAvailable())
        {
            // In terms of Gs
            IMU.readAcceleration(Ax, Ay, Az);
        }

        if (IMU.gyroscopeAvailable())
        {
            // In degree/sec
            IMU.readGyroscope(Gx, Gy, Gz);
        }

        AxError += Ax;
        AyError += Ay;
        AzError += Az;

        GxError += Gy;
        GyError += Gy;
        GzError += Gz;
    }

    AxError /= 1000;
    AyError /= 1000;
    AzError /= 1000;

    GxError /= 1000;
    GyError /= 1000;
    GzError /= 1000;

    AxPrev = Ax;
    AyPrev = Ay;
    AyPrev = Ay;

    GxPrev = Gx;
    GyPrev = Gy;
    GyPrev = Gy;
}

bool gyro::getData(gyroData &data)
{
    if (IMU.accelerationAvailable())
    {
        // In terms of Gs
        IMU.readAcceleration(Ax, Ay, Az);
    }
    else
    {
        return false;
    }

    if (IMU.gyroscopeAvailable())
    {
        // In degree/sec
        IMU.readGyroscope(Gx, Gy, Gz);
    }
    else
    {
        return false;
    }

    if (IMU.temperatureAvailable())
    {
        IMU.readTemperature(temperature_deg);
    }
    else
    {
        return false;
    }

    // Subtract the offset.
    Ax = Ax - AxError;
    Ay = Ay - AyError;
    Az = Az - AzError;

    Gx = Gx - GxError;
    Gy = Gy - GyError;
    Gz = Gz - GzError;

    // Add the low pass filter
    Ax = dampingFactor * AxPrev + (1 - dampingFactor) * Ax;
    Ay = dampingFactor * AyPrev + (1 - dampingFactor) * Ay;
    Az = dampingFactor * AzPrev + (1 - dampingFactor) * Az;

    Gx = dampingFactor * GxPrev + (1 - dampingFactor) * Gx;
    Gy = dampingFactor * GyPrev + (1 - dampingFactor) * Gy;
    Gz = dampingFactor * GzPrev + (1 - dampingFactor) * Gz;

    // Set the output values.
    data.Ax = Ax;
    data.Ay = Ay;
    data.Az = Az;

    data.Gx = Gx;
    data.Gy = Gy;
    data.Gz = Gz;

    data.temperature_deg = temperature_deg;

    // Set the previous values.
    AxPrev = Ax;
    AyPrev = Ay;
    AzPrev = Az;

    GxPrev = Gx;
    GyPrev = Gy;
    GzPrev = Gz;

    return true;
}