import numpy as np
from numpy.linalg import inv, norm
from numpy import asarray, matrix
from math import *
import math
#import matplotlib.pyplot as plt


_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

_NEXT_AXIS = [1, 2, 0, 1]

def euler2mat(ai, aj, ak, axes='sxyz'):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array (3, 3)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M

def point_to_rad(p1, p2): # converts 2D cartesian points to polar angles in range 0 - 2pi
    theta = atan2(p2, p1)
    theta = (theta + 2*pi) % (2*pi)
    return theta
    if (p1 > 0 and p2 >= 0): return atan(p2/(p1))
    elif (p1 == 0 and p2 >= 0): return pi/2
    elif (p1 < 0 and p2 >= 0): return -abs(atan(p2/p1)) + pi
    elif (p1 < 0 and p2 < 0): return atan(p2/p1) + pi
    elif (p1 > 0 and p2 < 0): return -abs(atan(p2/p1)) + 2*pi
    elif (p1 == 0 and p2 < 0): return pi * 3/2
    elif (p1 == 0 and p2 == 0): return pi * 3/2 # edge case


def RotMatrix3D(rotation=[0,0,0],is_radians=True, order='xyz'):
    
    roll, pitch, yaw = rotation[0], rotation[1], rotation[2]

    # convert to radians is the input is in degrees
    if not is_radians: 
        roll = radians(roll)
        pitch = radians(pitch)
        yaw = radians(yaw)
    
    # rotation matrix about each axis
    rotX = np.matrix([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
    rotY = np.matrix([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
    rotZ = np.matrix([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
    
    # rotation matrix order (default: pitch -> roll -> yaw)
    if order == 'xyz': rotationMatrix = rotZ * rotY * rotX
    elif order == 'xzy': rotationMatrix = rotY * rotZ * rotX
    elif order == 'yxz': rotationMatrix = rotZ * rotX * rotY
    elif order == 'yzx': rotationMatrix = rotX * rotZ * rotY
    elif order == 'zxy': rotationMatrix = rotY * rotX * rotZ
    elif order == 'zyx': rotationMatrix = rotX * rotY * rotZ
    
    return rotationMatrix # roll pitch and yaw rotation 


def leg_explicit_inverse_kinematics(r_body_foot, leg_index, config):
    """Find the joint angles corresponding to the given body-relative foot position for a given leg and configuration
    
    Parameters
    ----------
    r_body_foot : numpy array (3)
        The x,y,z co-ordinates of the foot relative to the first leg frame
    leg_index : int
        The index of the leg (0-3), which represents which leg it is. 0 = Front left, 1 = Front right, 2 = Rear left, 3 = Rear right
    config : Configuration class
        Configuration class which contains all of the parameters of the Dingo (link lengths, max velocities, etc)
    
    Returns
    -------
    angles : numpy array (3)
        Array of calculated joint angles (theta_1, theta_2, theta_3) for the input position
    """

    #Determine if leg is a right or a left leg
    if leg_index == 1 or leg_index == 3:
        is_right = 0
    else:
        is_right = 1
    
    #Flip the y axis if the foot is a right foot to make calculation correct
    x,y,z = r_body_foot[0], r_body_foot[1], r_body_foot[2]
    if is_right: y = -y

    r_body_foot = np.array([x,y,z])
    
    #rotate the origin frame to be in-line with config.L1 for calculating theta_1 (rotation about x-axis):
    R1 = pi/2 - config.phi 
    rot_mtx = RotMatrix3D([-R1,0,0],is_radians=True)
    r_body_foot_ = rot_mtx * (np.reshape(r_body_foot,[3,1]))
    r_body_foot_ = np.ravel(r_body_foot_)
    
    # xyz in the rotated coordinate system
    x = r_body_foot_[0]
    y = r_body_foot_[1]
    z = r_body_foot_[2]

    # length of vector projected on the YZ plane. equiv. to len_A = sqrt(y**2 + z**2)
    len_A = norm([0,y,z])   
    # a_1 : angle from the positive y-axis to the end-effector (0 <= a_1 < 2pi)
    # a_2 : angle bewtween len_A and leg's projection line on YZ plane
    # a_3 : angle between link1 and length len_A
    a_1 = point_to_rad(y,z)                     
    a_2 = asin(sin(config.phi)*config.L1/len_A)
    a_3 = pi - a_2 - config.phi               

    # angle of link1 about the x-axis 
    if is_right: theta_1 = a_1 + a_3
    else: 
        theta_1 = a_1 + a_3
    if theta_1 >= 2*pi: theta_1 = np.mod(theta_1,2*pi)
    
    #Translate frame to the frame of the leg
    offset = np.array([0.0,config.L1*cos(theta_1),config.L1*sin(theta_1)])
    translated_frame = r_body_foot_ - offset
    
    if is_right: R2 = theta_1 + config.phi - pi/2
    else: R2 = -(pi/2 - config.phi + theta_1) #This line may need to be adjusted
    R2 = theta_1 + config.phi - pi/2

    # create rotation matrix to work on a new 2D plane (XZ_)
    rot_mtx = RotMatrix3D([-R2,0,0],is_radians=True)
    j4_2_vec_ = rot_mtx * (np.reshape(translated_frame,[3,1]))
    j4_2_vec_ = np.ravel(j4_2_vec_)
    
    # xyz in the rotated coordinate system + offset due to link_1 removed
    x_, y_, z_ = j4_2_vec_[0], j4_2_vec_[1], j4_2_vec_[2]
    
    len_B = norm([x_, 0, z_])
    
    # handling mathematically invalid input, i.e., point too far away to reach
    if len_B >= (config.L2 + config.L3): 
        len_B = (config.L2 + config.L3) * 0.8
        print('target coordinate: [%f %f %f] too far away', x, y, z)
    
    # b_1 : angle between +ve x-axis and len_B (0 <= b_1 < 2pi)
    # b_2 : angle between len_B and link_2
    # b_3 : angle between link_2 and link_3
    b_1 = point_to_rad(x_, z_)  
    b_2 = acos((config.L2**2 + len_B**2 - config.L3**2) / (2 * config.L2 * len_B)) 
    b_3 = acos((config.L2**2 + config.L3**2 - len_B**2) / (2 * config.L2 * config.L3))  
    
    theta_2 = b_1 - b_2
    theta_3 = pi - b_3

    # modify angles to match robot's configuration (i.e., adding offsets)
    angles = angle_corrector(angles=[theta_1, theta_2, theta_3])
    return np.array(angles)



def four_legs_inverse_kinematics(r_body_foot, config):
    """Find the joint angles for all twelve DOF correspoinding to the given matrix of body-relative foot positions.
    
    Parameters
    ----------
    r_body_foot : numpy array (3,4)
        Matrix of the body-frame foot positions. Each column corresponds to a separate foot.
    config : Config object
        Object of robot configuration parameters.
    
    Returns
    -------
    numpy array (3,4)
        Matrix of corresponding joint angles.
    """
    # print('r_body_foot: \n',np.round(r_body_foot,3))
    alpha = np.zeros((3, 4))
    for i in range(4):
        body_offset = config.LEG_ORIGINS[:, i]
        alpha[:, i] = leg_explicit_inverse_kinematics(
            r_body_foot[:, i] - body_offset, i, config
        )
    return alpha #[Front Right, Front Left, Back Right, Back Left]

def forward_kinematics(angles, config, is_right = 0):
    """Find the foot position corresponding to the given joint angles for a given leg and configuration
    
    Parameters
    ----------
    angles : numpy array (3)
        desired joint angles: theta1, theta2, theta3 
    config : Configuration class
        Configuration class which contains all of the parameters of the Dingo (link lengths, max velocities, etc)
    is_right : int
        An integer indicating whether the leg is a left or right leg
    
    Returns
    -------
    angles : numpy array (3)
        Array of corresponding task space values (x,y,z) relative to the base frame of each leg
    """
    x = config.L3*sin(angles[1]+angles[2]) - config.L2*cos(angles[1])
    y = 0.5*config.L2*cos(angles[0]+angles[1]) - config.L1*cos(angles[0]+(403*pi)/4500) - 0.5*config.L2*cos(angles[0]-angles[1]) - config.L3*cos(angles[1]+angles[2])*sin(angles[0])
    z = 0.5*config.L2*sin(angles[0]-angles[1]) + config.L1*sin(angles[0]+(403*pi)/4500) - 0.5*config.L2*sin(angles[0]+angles[1]) - config.L3*cos(angles[1]+angles[2])*cos(angles[0])
    if not is_right:
        y = -y
    return np.array([x,y,z])

def angle_corrector(angles=[0,0,0]):
    # assuming theta_2 = 0 when the leg is pointing down (i.e., 270 degrees offset from the +ve x-axis)
    angles[0] = angles[0]
    angles[1] = angles[1] - pi #theta2 offset
    angles[2] = angles[2] - pi/2 #theta3 offset

    #Adjusting for angles out of range, and making angles be between -pi,pi
    for index, theta in enumerate(angles):
        if theta > 2*pi: angles[index] = np.mod(theta,2*pi)
        if theta > pi: angles[index] = -(2*pi - theta)
    return angles


class Configuration:
    def __init__(self):
        #################### COMMANDS ####################
        self.max_x_velocity = 1.2
        self.max_y_velocity = 0.5
        self.max_yaw_rate = 2.0
        self.max_pitch = 30.0 * np.pi / 180.0
        
        #################### MOVEMENT PARAMS ####################
        self.z_time_constant = 0.02
        self.z_speed = 0.06  # maximum speed [m/s]
        self.pitch_deadband = 0.05
        self.pitch_time_constant = 0.25
        self.max_pitch_rate = 0.3
        self.roll_speed = 0.1  # maximum roll rate [rad/s]
        self.yaw_time_constant = 0.3
        self.max_stance_yaw = 1.2
        self.max_stance_yaw_rate = 1

        #################### STANCE ####################
        self.delta_x = 0.117 #- 0.00535 #115650.00535

        #These x_shift variables will move the default foot positions of the robot 
        #Handy if the centre of mass shifts as can move the feet to compensate
        self.rear_leg_x_shift = -0.04 #In default config, the robots mass is slightly biased to the back feet, so the back feet are shifted back slightly
        self.front_leg_x_shift = 0.00

        self.delta_y = 0.1106 #0.1083
        self.default_z_ref = -0.25 #-0.16

        #################### SWING ######################
        self.z_coeffs = None
        self.z_clearance = 0.07
        self.alpha = (
            0.5  # Ratio between touchdown distance and total horizontal stance movement
        )
        self.beta = (
            0.5  # Ratio between touchdown distance and total horizontal stance movement
        )

        #################### GAIT #######################
        self.dt = 0.01
        self.num_phases = 4
        self.contact_phases = np.array(
            [[1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0]]
        )
        self.overlap_time = (
            0.04  # duration of the phase where all four feet are on the ground
        )
        self.swing_time = (
            0.07  # duration of the phase when only two feet are on the ground
        )

        ######################## GEOMETRY ######################
        self.LEG_FB = 0.11165 #   front-back distance from center line to leg axis
        self.LEG_LR = 0.061  # left-right distance from center line to leg plane
        self.LEG_ORIGINS = np.array( #Origins of the initial frame from the centre of the body
            [
                [self.LEG_FB, self.LEG_FB, -self.LEG_FB, -self.LEG_FB],
                [-self.LEG_LR, self.LEG_LR, -self.LEG_LR, self.LEG_LR],
                [0, 0, 0, 0],
            ]
        )

        #np.array([[self.LEG_FB, self.LEG_FB, -self.LEG_FB, -self.LEG_FB],[-self.LEG_LR, self.LEG_LR, -self.LEG_LR, self.LEG_LR],[0, 0, 0, 0],])

        #Leg lengths
        self.L1 = 0.05162024721
        self.L2 = 0.130
        self.L3 = 0.13813664159
        self.phi = math.radians(73.91738698)
        
        ################### INERTIAL ####################
        self.FRAME_MASS = 0.560  # kg
        self.MODULE_MASS = 0.080  # kg
        self.LEG_MASS = 0.030  # kg
        self.MASS = self.FRAME_MASS + (self.MODULE_MASS + self.LEG_MASS) * 4

        # Compensation factor of 3 because the inertia measurement was just
        # of the carbon fiber and plastic parts of the frame and did not
        # include the hip servos and electronics
        self.FRAME_INERTIA = tuple(
            map(lambda x: 3.0 * x, (1.844e-4, 1.254e-3, 1.337e-3))
        )
        self.MODULE_INERTIA = (3.698e-5, 7.127e-6, 4.075e-5)

        leg_z = 1e-6
        leg_mass = 0.010
        leg_x = 1 / 12 * self.L2 ** 2 * leg_mass
        leg_y = leg_x
        self.LEG_INERTIA = (leg_x, leg_y, leg_z)

    @property
    def default_stance(self): #Default stance of the robot relative to the centre frame
        return np.array(
            [
                [
                    self.delta_x + self.front_leg_x_shift, #Front Right
                    self.delta_x + self.front_leg_x_shift, #Front Left
                    -self.delta_x + self.rear_leg_x_shift, #Back Right
                    -self.delta_x + self.rear_leg_x_shift, #Back Left
                ],
                [-self.delta_y, self.delta_y, -self.delta_y, self.delta_y],
                [0, 0, 0, 0],
            ]
        )

    ################## SWING ###########################
    @property
    def z_clearance(self):
        return self.__z_clearance

    @z_clearance.setter
    def z_clearance(self, z):
        self.__z_clearance = z
        # b_z = np.array([0, 0, 0, 0, self.__z_clearance])
        # A_z = np.array(
        #     [
        #         [0, 0, 0, 0, 1],
        #         [1, 1, 1, 1, 1],
        #         [0, 0, 0, 1, 0],
        #         [4, 3, 2, 1, 0],
        #         [0.5 ** 4, 0.5 ** 3, 0.5 ** 2, 0.5 ** 1, 0.5 ** 0],
        #     ]
        # )   
        # self.z_coeffs = solve(A_z, b_z)

    ########################### GAIT ####################
    @property
    def overlap_ticks(self):
        return int(self.overlap_time / self.dt)

    @property
    def swing_ticks(self):
        return int(self.swing_time / self.dt)

    @property
    def stance_ticks(self):
        return 2 * self.overlap_ticks + self.swing_ticks

    @property
    def phase_ticks(self):
        return np.array(
            [self.overlap_ticks, self.swing_ticks, self.overlap_ticks, self.swing_ticks]
        )

    @property
    def phase_length(self):
        return 2 * self.overlap_ticks + 2 * self.swing_ticks

def convert_radians_to_degrees(radians_list):
    return [[num*(180/pi) for num in inner_list] for inner_list in radians_list]

if __name__ == "__main__":
    config = Configuration()

    foot_locations = [
    [0, 0, 0, 0],
    [-0.22, 0, 0, 0],
    [0.049, 0, 0, 0]
    ]
    data = four_legs_inverse_kinematics(np.array(foot_locations), config)
    data = convert_radians_to_degrees(data)
    print(data[0][0], data[1][0], data[2][0])
    