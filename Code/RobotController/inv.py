import time
from lsm6dsox import LSM6DSOX
from machine import Pin, PWM, I2C
import bluetooth
import time
from machine import Pin
from math import *


class Configuration:
    def __init__(self):
        self.delta_x = 0.117
        self.rear_leg_x_shift = -0.04
        self.front_leg_x_shift = 0.00
        self.delta_y = 0.1106
        self.default_z_ref = -0.25

        self.alpha = 0.5
        self.beta = 0.5

        self.LEG_FB = 0.11165
        self.LEG_LR = 0.061
        self.LEG_ORIGINS = [
            [self.LEG_FB, self.LEG_FB, -self.LEG_FB, -self.LEG_FB],
            [-self.LEG_LR, self.LEG_LR, -self.LEG_LR, self.LEG_LR],
            [0, 0, 0, 0],
        ]

        self.L1 = 0.05162024721
        self.L2 = 0.130
        self.L3 = 0.13813664159
        self.phi = radians(73.91738698)
        
        
rightHip = "RIGHTHIP"
rightHip2 = "RIGHTHIP2"
rightKnee = "RIGHTKNEE"

leftHip = "LEFTHIP"
leftHip2 = "LEFTHIP2"
leftKnee = "LEFTKNEE"

class JointPosition:
    def __init__(self):
        self.data = dict()
        self.data[rightHip] = 88
        self.data[rightHip2] = 90
        self.data[rightKnee] = 90
        
        self.data[leftHip] = 83
        self.data[leftHip2] = 90
        self.data[leftKnee] = 90
        
    def setJointAngle(self, key, value):
        '''
        limits = self._jointLimits(key)
        if limits == None:
            return
        if limits[1] < value:
            self.data[key] = limits[1]
            return
        if limits[0] > value:
            self.data[key] = limits[0]
            return
        '''
        self.data[key] = value
        
    def _jointLimits(self, key):
        if key == rightHip or key == leftHip:
            return (45, 135)
        if key == rightHip2 or key == leftHip2:
            return (30, 150)
        if key == rightKnee or key == leftKnee:
            return (30, 150)
        return None
    
    def equals(self, other, threshold=1e-2):
        for key in self.data:
            if key not in other.data:
                return False
            if abs(self.data[key] - other.data[key]) > threshold:
                return False
        return True
    
    
    def add(self, other):
        result = JointPosition()
        for key in self.data:
            if key in other.data:
                result.setJointAngle(key, self.data[key] + other.data[key])
        return result
    
class Servo:
    __servo_pwm_freq = 50
    __min_u16_duty = 1802
    __max_u16_duty = 7864
    min_angle = 0
    max_angle = 180
    current_angle = 0.001


    def __init__(self, pin):
        self.__initialise(pin)


    def update_settings(self, servo_pwm_freq, min_u16_duty, max_u16_duty, min_angle, max_angle, pin):
        self.__servo_pwm_freq = servo_pwm_freq
        self.__min_u16_duty = min_u16_duty
        self.__max_u16_duty = max_u16_duty
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.__initialise(pin)


    def move(self, angle):
        # round to 2 decimal places, so we have a chance of reducing unwanted servo adjustments
        angle = round(angle, 2)
        # do we need to move?
        if angle == self.current_angle:
            return
        self.current_angle = angle
        # calculate the new duty cycle and move the motor
        duty_u16 = self.__angle_to_u16_duty(angle)
        self.__motor.duty_u16(duty_u16)
    
    def stop(self):
        self.__motor.deinit()
    
    def get_current_angle(self):
        return self.current_angle

    def __angle_to_u16_duty(self, angle):
        return int((angle - self.min_angle) * self.__angle_conversion_factor) + self.__min_u16_duty


    def __initialise(self, pin):
        self.current_angle = -0.001
        self.__angle_conversion_factor = (self.__max_u16_duty - self.__min_u16_duty) / (self.max_angle - self.min_angle)
        self.__motor = PWM(Pin(pin))
        self.__motor.freq(self.__servo_pwm_freq)



class Robot:
    def __init__(self):
        self.currentState = JointPosition()
        self.desiriedState = JointPosition()
        self.nextStep = JointPosition()
        
        self.calculated = False
        self.speed = 0.01
        
        self.joints = dict()
        
        ## Init the servo pins
        self.joints[rightHip] = Servo(pin=25)
        self.joints[rightHip2] = Servo(pin=15)
        self.joints[rightKnee] = Servo(pin=16)
        
        self.joints[leftHip] = Servo(pin=17)
        self.joints[leftHip2] = Servo(pin=18)
        self.joints[leftKnee] = Servo(pin=19)
        
    def setNextState(self, jointPosition):
        self.calculated = False
        self.desiriedState = jointPosition
        self.calculateNextSteps()
    
    def calculateNextSteps(self, numOfSteps = 1000):
        if self.desiriedState.equals(self.currentState):
            self.currentState = desiriedState
            return self.desiriedState
        
        for key in self.currentState.data:
            currentAngle = self.currentState.data[key]
            expectedAngle = self.desiriedState.data[key]
                        
            self.nextStep.setJointAngle(key, (expectedAngle-currentAngle)/numOfSteps)

                    
        print(self.nextStep.data)
        self.calculated = True
        
    def getNextStep(self):
        if self.calculated:
            if self.currentState.equals(self.desiriedState):
                self.calculated = False
                return None
            data = self.currentState.add(self.nextStep)
            self.currentState = data
            return data
        return None
    
    def move(self, jointData=None):
        if jointData != None:
            for key in jointData.data:
                self.joints[key].move(jointData.data[key])
        
    def moveToNextStep(self):
        self.move(self.getNextStep())


    def _getJointDataForXYZRight(self, x, y, z, jointData):
        alpha, beta, gamma = self._moveToPos(x, y, z)
        jointData.setJointAngle(rightHip, alpha)
        jointData.setJointAngle(rightHip2, beta)
        jointData.setJointAngle(rightKnee, gamma)

    def _getJointDataForXYZLeft(self, x, y, z, jointData):
        alpha, beta, gamma = self._moveToPos(x, y, z)
        jointData.setJointAngle(leftHip, alpha)
        jointData.setJointAngle(leftHip2, beta)
        jointData.setJointAngle(leftKnee, gamma)

def r2ds(radians_list):
    return [[num*(180/pi) for num in inner_list] for inner_list in radians_list]

def r2d(num):
    return num*(180/pi)

def invK(x, y, z, jointData):
    L2 = 150
    L3 = 150
    
    h = sqrt(y**2 + z**2)
    
    if (h > L2 + L3 and h < 50):
        return jointData
    
    theta = atan2(y, z)
    tita = acos((L2**2 + h**2 - L3**2) / (2*L2*h))

    a2 = theta + tita
    a3 = theta - tita
    
    a2 = r2d(a2) + 45
    a3 = r2d(a3) + 45
    
    
    print(a2, a3)
    
    jointData.setJointAngle(rightHip2, a2)
    #jointData.setJointAngle(rightKnee, a3)
    
    return jointData
    
robot = Robot()
jointData = JointPosition()

jointData = invK(0, 0, 0, jointData)

robot.move(jointData)