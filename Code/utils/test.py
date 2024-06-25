import pybullet as p
import time
import pybullet_data

def print_link_ids(robot):
    num_joints = p.getNumJoints(robot)
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot, joint_index)
        link_name = joint_info[12].decode('utf-8')
        print(f"Link ID: {joint_index}, Link Name: {link_name}")

# Example usage
p.connect(p.GUI)  # Use DIRECT mode to avoid opening a GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("../urdf/robot.urdf", [0, 0, 0.25], [0, 0, 0, 1])

try:
    while 1:
        p.stepSimulation()
        current_position, current_orientation = p.getBasePositionAndOrientation(robot)
        print(current_position)
        time.sleep(0.01)
except KeyboardInterrupt:
    pass

print_link_ids(robot)

p.disconnect()