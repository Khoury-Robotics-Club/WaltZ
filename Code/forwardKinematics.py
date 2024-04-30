from math import sin, cos, acos, sqrt, pi

# Lengths of the links
L1 = 25
L2 = 150
L3 = 150

def forwardKinematics(theta1, theta2, theta3):
    y = L3*sin(theta3-theta2) + L2*sin(theta2)
    x = L2*cos(theta3-theta2)-L3*cos(theta2)
    t = sqrt(L1*2 + y*2)
    z = t*cos(pi/2-acos(y/t)-theta1)
    
    return x,y,z

## Sanity testing code.
if __name__ == "__main__":
    print(forwardKinematics(0, pi/2, pi))