import time
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

config = Configuration()


d = forward_kinematics(list(map(lambda x: x*(pi/180), [90, 90, 90])), config, 0)
print(d)


data = leg_explicit_inverse_kinematics(d, 0, config)

data = list(map(lambda x: x*(180/pi), data))

print(data)