import tensorflow as tf
import math

def Fermi(x):
        return 1.0 / (tf.math.exp(x) + 1.0)

def AbsSin(x):
        return tf.math.abs(tf.math.sin(x * math.pi /2))

def IsFloat(x):
        floatNumbers = x - tf.floor(x)
        return tf.cast(floatNumbers > 0, tf.float64), tf.cast(floatNumbers <= 0, tf.float64)