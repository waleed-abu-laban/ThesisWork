import tensorflow as tf
import math

class LossFunctionNULL(object): # no loss function
    def __call__(self, decodedLLRs):
        return tf.zeros_like(decodedLLRs)

    def SetLabels(self, channelLabels):
        None

    def SetHOrtho(self, Hortho):
        None

class LossFunctionLiu(object): # Liu paper loss function
    def __call__(self, decodedLLRs):
        return tf.reduce_mean(self.__AbsSin(self.Hortho@(self.labels + self.__Fermi(decodedLLRs))), axis = 0) #/ self.lMax
    
    def SetLabels(self, channelLabels):
        self.labels = tf.constant(channelLabels, tf.float64)

    def SetHOrtho(self, Hortho):
        self.Hortho = Hortho

    def __Fermi(self, x):
        return 1.0 / (tf.math.exp(x) + 1.0)

    def __AbsSin(self, x):
        return tf.math.abs(tf.math.sin(x * math.pi /2))

class LossFunctionCE(object): # Cross Entropy loss function
    def __call__(self, decodedLLRs):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-decodedLLRs, labels=self.labels), axis=0) #/ lMax
        
    def SetHOrtho(self, Hortho):
        None

    def SetLabels(self, channelLabels):
        self.labels = tf.constant(channelLabels, tf.float64)



        