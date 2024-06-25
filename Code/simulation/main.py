import pybullet as p
import pybullet_data
import numpy as np
import time


p.connect(p.GUI) #p.DIRECT (for training on server)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)


p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1]) ## [X,Y,Z] and quaternion
robot = p.loadURDF("../urdf/robot.urdf", [0, 0, 0.25], [0, 0, 0, 1]) # useFixedBase = True, set this for robot arm
objOfFocus = robot

for step in range(10000):
    focusPos, _ = p.getBasePositionAndOrientation(robot)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focusPos)
    p.stepSimulation()
    time.sleep(0.01)


