import time
from lsm6dsox import LSM6DSOX
from machine import Pin, PWM, I2C, UART
import random
import struct
import time
from machine import Pin
from micropython import const
import math

from math import *

def dot_product(A, B):
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def matrix_multiplication(A, B):
    return dot_product(A, B)

def point_to_rad(p1, p2):
    theta = atan2(p2, p1)
    theta = (theta + 2 * pi) % (2 * pi)
    return theta

def RotMatrix3D(rotation=[0, 0, 0], is_radians=True, order='xyz'):
    roll, pitch, yaw = rotation[0], rotation[1], rotation[2]

    if not is_radians:
        roll = radians(roll)
        pitch = radians(pitch)
        yaw = radians(yaw)

    rotX = [[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]]
    rotY = [[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]]
    rotZ = [[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]]

    if order == 'xyz':
        rotationMatrix = matrix_multiplication(matrix_multiplication(rotZ, rotY), rotX)
    return rotationMatrix

def leg_explicit_inverse_kinematics(r_body_foot, leg_index, config):
    if leg_index == 1 or leg_index == 3:
        is_right = 0
    else:
        is_right = 1

    x, y, z = r_body_foot[0], r_body_foot[1], r_body_foot[2]
    if is_right: 
        y = -y

    R1 = pi / 2 - config.phi 
    rot_mtx = RotMatrix3D([-R1, 0, 0], is_radians=True)
    r_body_foot_ = [sum(a * b for a, b in zip(rot_mtx_row, [x, y, z])) for rot_mtx_row in rot_mtx]

    x, y, z = r_body_foot_
    len_A = sqrt(y**2 + z**2)
    a_1 = point_to_rad(y, z)                     
    a_2 = asin(sin(config.phi) * config.L1 / len_A)
    a_3 = pi - a_2 - config.phi

    theta_1 = a_1 + a_3
    if theta_1 >= 2 * pi: 
        theta_1 %= 2 * pi
    
    offset = [0.0, config.L1 * cos(theta_1), config.L1 * sin(theta_1)]
    translated_frame = [r_body_foot_[i] - offset[i] for i in range(3)]

    if is_right: 
        R2 = theta_1 + config.phi - pi / 2
    else: 
        R2 = -(pi / 2 - config.phi + theta_1)
    R2 = theta_1 + config.phi - pi / 2

    rot_mtx = RotMatrix3D([-R2, 0, 0], is_radians=True)
    j4_2_vec_ = [sum(a * b for a, b in zip(rot_mtx_row, translated_frame)) for rot_mtx_row in rot_mtx]
    
    x_, y_, z_ = j4_2_vec_
    len_B = sqrt(x_**2 + z_**2)

    if len_B >= (config.L2 + config.L3): 
        len_B = (config.L2 + config.L3) * 0.8
        print('target coordinate: [%f %f %f] too far away', x, y, z)
    
    b_1 = point_to_rad(x_, z_)
    b_2 = acos((config.L2**2 + len_B**2 - config.L3**2) / (2 * config.L2 * len_B)) 
    b_3 = acos((config.L2**2 + config.L3**2 - len_B**2) / (2 * config.L2 * config.L3))  
    
    theta_2 = b_1 - b_2
    theta_3 = pi - b_3

    angles = angle_corrector([theta_1, theta_2, theta_3])
    return angles

def four_legs_inverse_kinematics(r_body_foot, config):
    alpha = [[0 for _ in range(4)] for _ in range(3)]
    for i in range(4):
        #body_offset = config.LEG_ORIGINS[i]
        body_offset = [row[i] for row in config.LEG_ORIGINS]

        angles = leg_explicit_inverse_kinematics(
            [r_body_foot[j][i] - body_offset[j] for j in range(3)], i, config
        )
        for j in range(3):
            alpha[j][i] = angles[j]
    return alpha

def forward_kinematics(angles, config, is_right=0):
    x = config.L3 * sin(angles[1] + angles[2]) - config.L2 * cos(angles[1])
    y = 0.5 * config.L2 * cos(angles[0] + angles[1]) - config.L1 * cos(angles[0] + (403 * pi) / 4500) - 0.5 * config.L2 * cos(angles[0] - angles[1]) - config.L3 * cos(angles[1] + angles[2]) * sin(angles[0])
    z = 0.5 * config.L2 * sin(angles[0] - angles[1]) + config.L1 * sin(angles[0] + (403 * pi) / 4500) - 0.5 * config.L2 * sin(angles[0] + angles[1]) - config.L3 * cos(angles[1] + angles[2]) * cos(angles[0])
    if not is_right:
        y = -y
    return [x, y, z]

def angle_corrector(angles=[0, 0, 0]):
    angles[1] -= pi
    angles[2] -= pi / 2

    for index, theta in enumerate(angles):
        if theta > 2 * pi:
            angles[index] %= 2 * pi
        if theta > pi:
            angles[index] = -(2 * pi - theta)
    return angles

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


def convert_radians_to_degrees(radians_list):
    return [[num*(180/pi) for num in inner_list] for inner_list in radians_list]


LED_PIN = 6

## Constants
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
    
    def calculateNextSteps(self, numOfSteps = 100):
        if self.desiriedState.equals(self.currentState):
            self.currentState = self.desiriedState
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
        for _ in range(100):
            self.move(self.getNextStep())
            time.sleep_ms(20)


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
    
resetPos = JointPosition()

lsm = LSM6DSOX(I2C(0, scl=Pin(13), sda=Pin(12)))
robot = Robot()

config = Configuration()
jointData = JointPosition()

robot.move(resetPos)
time.sleep_ms(1000)

#jointData.setJointAngle(rightHip, 90)
jointData.setJointAngle(rightHip2, 110)
jointData.setJointAngle(rightKnee, 70)

#jointData.setJointAngle(leftHip, 90)
jointData.setJointAngle(leftHip2, 70)
jointData.setJointAngle(leftKnee, 110)
robot.setNextState(jointData)


robot.moveToNextStep()
time.sleep_ms(2000)

uart = UART(0, baudrate=9600, tx=Pin(0), rx=Pin(1))

def send_bluetooth(data):
    uart.write(data)
    
def receive_bluetooth():
    if uart.any():
        return uart.read().decode('utf-8')
    return None


robot.move(resetPos)

while (True):
    data = receive_bluetooth()
    if data:
        print("Received:", data)
        # Echo the data back
        send_bluetooth("Echo: " + data + "\n")
    print('Accelerometer: x:{:>8.3f} y:{:>8.3f} z:{:>8.3f}'.format(*lsm.read_accel()))
    print('Gyroscope:     x:{:>8.3f} y:{:>8.3f} z:{:>8.3f}'.format(*lsm.read_gyro()))
    print("")
    time.sleep_ms(100)