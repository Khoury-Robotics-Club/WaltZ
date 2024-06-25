import numpy as np
from math import pi

def convertToRadians(data:float) -> float:
    return data*(pi/180)


def convertToDegree(data:float) -> float:
    return data*(180/pi)


def convertToRadians(data:np.ndarray) -> np.ndarray:
    return data*(pi/180)

def convertToDegree(data:np.ndarray) -> np.ndarray:
    return data*(180/pi)

