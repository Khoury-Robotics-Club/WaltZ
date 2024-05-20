import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,10]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("./waltz.urdf.xml", startPos, startOrientation)

joint_name = "revoluteRightHip"  # Replace with the actual joint name
joint_index = p.getJointInfo(boxId, p.JOINT_REVOLUTE)[0]  # Get the joint index

# Set joint motor control (e.g., velocity control with zero target velocity for joint friction)
p.setJointMotorControl2(boxId, joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=10, force=0)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
