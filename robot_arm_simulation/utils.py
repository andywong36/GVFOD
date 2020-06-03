import math


def get_angle(a, b, c):
    """ Cosine law """
    from math import acos
    C = acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    return C


def tanh(x, scale=1):
    return math.tanh(x / scale)


def dtanh(x, scale=1):
    return (1 - tanh(x, scale) ** 2) / scale

def ssign(x):
    """ A smooth sign function """
    return tanh(x, scale=0.05)

def dssign(x):
    """ Derivative of the smooth sign function"""
    return dtanh(x, scale=0.05)