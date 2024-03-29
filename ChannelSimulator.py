import numpy as np
import tensorflow as tf

def SimulateChannel(y, channelType, parameter = 0, codeRate = 1):
    data = 0 * y
    if(channelType == 'BSC'):
        flipVec = 1*(np.random.rand(y.size) <= parameter)  # eps = parameter
        data = 1*np.logical_xor(y, flipVec)
    elif(channelType == 'AWGN'):
        # Symbol mapper
        symbols = y * -2 +1
        # Noise
        EbN0 = 10**(parameter/10) # EbN0dB = parameter
        No = 1/(EbN0 * codeRate)
        noiseVar = No/2
        noise = np.random.randn(symbols.size) * np.sqrt(noiseVar)
        # Modulated data
        data = symbols + noise
    return data

def CalculateLLRs(y, channelType, parameter = 0, codeRate = 1):
    LLRs = y * 0
    if(channelType == 'BSC'):
        y1LLR = np.log(parameter) - np.log(1-parameter) # eps = parameter
        LLRs = (y*2-1) * y1LLR
    elif(channelType == 'AWGN'):
        EbN0 = 10**(parameter/10) # EbN0dB = parameter
        No = 1/(EbN0 * codeRate)
        noiseVar = No/2
        LLRs = y*2 / noiseVar
    return LLRs

class Channel(object):
    def __init__(self, N, batchSize, channelType, parameters, codeRate, isQuantum, generatorMatrix):
        self.N = N
        self.batchSize = batchSize
        self.channelType = channelType
        self.parameters = parameters
        self.codeRate = codeRate
        self.isQuantum = isQuantum
        self.generatorMatrix = generatorMatrix
        np.random.seed(100)

    def GetBatchSize(self):
        return self.batchSize

    def SetBatchSize(self, batchSize):
        self.batchSize = batchSize

    def SetChannelParameters(self, parameters):
        self.parameters = parameters

    def SimulateChannelTF(self, y):
        sectionsCount = len(self.parameters)
        reshapedY = np.reshape(y, [sectionsCount, self.batchSize * self.N // sectionsCount], order='F')
        data = 0 * reshapedY
        dataShape = data.shape
        if(self.channelType == 'BSC'):
            flipVec = 1*(np.random.rand(dataShape[0], dataShape[1]) <= np.repeat(np.expand_dims(self.parameters, -1), dataShape[1], axis = 1)) # eps = parameter
            # if(self.isQuantum): # Add the Y error
            #     flipVecY = 1*(np.random.rand(dataShape[0], dataShape[1]//2) <= np.repeat(np.expand_dims(self.parameters, -1), dataShape[1]//2, axis = 1)) # eps = parameter
            #     flipVecY = np.tile(flipVecY, 2)
            #     flipVec = (flipVecY + flipVec) % 2
            data = 1*np.logical_xor(reshapedY, flipVec)
        elif(self.channelType == 'AWGN'):
            # Symbol mapper
            symbols = reshapedY * -2 +1
            # Noise
            EbN0 = 10**(self.parameters/10) # EbN0dB = parameter
            No = 1/(EbN0 * self.codeRate)
            noiseVar = No/2
            noise = np.random.randn(dataShape[0], dataShape[1]) * np.repeat(np.expand_dims(np.sqrt(noiseVar), -1), dataShape[1], axis = 1)
            # Modulated data
            data = symbols + noise
        return np.reshape(data, [self.batchSize, self.N]).transpose()

    def CalculateLLRsTF(self, y):
        sectionsCount = len(self.parameters)
        reshapedY = np.reshape(y.transpose(), [sectionsCount, self.batchSize * self.N // sectionsCount])
        LLRsTemp =  0 * reshapedY
        if(self.channelType == 'BSC'):
            y1LLR = np.log(np.array(self.parameters)) - np.log(1-np.array(self.parameters)) # eps = parameter
            LLRsTemp = (reshapedY*2-1) * np.repeat(np.expand_dims(y1LLR, -1), reshapedY.shape[1], axis = 1) 
        elif(self.channelType == 'AWGN'):
            EbN0 = 10**(self.parameters/10) # EbN0dB = parameter
            No = 1/(EbN0 * self.codeRate)
            noiseVar = No/2
            LLRsTemp = reshapedY*2 / np.repeat(np.expand_dims(noiseVar, -1), reshapedY.shape[1], axis = 1)
        return tf.constant(np.reshape(LLRsTemp, [self.batchSize, self.N]).transpose(), dtype=tf.float64)

    def CalculateSingleLLRTF(self, y):
        if(self.channelType == 'BSC'):
            y1LLR = np.log(np.array(self.parameters)) - np.log(1-np.array(self.parameters)) # eps = parameter
            LLR = (y*2-1) * y1LLR 
        elif(self.channelType == 'AWGN'):
            EbN0 = 10**(self.parameters/10) # EbN0dB = parameter
            No = 1/(EbN0 * self.codeRate)
            noiseVar = No/2
            LLR = y*2 / noiseVar
        return LLR

    def GenerateLLRs(self, simulateChannel):
        if(simulateChannel):
            kWords = np.random.binomial(n=1, p=0.5, size=[self.batchSize, self.generatorMatrix.shape[0]])
            #testWord = np.array([ 1,     1,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]) #np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #
            channelInput = np.transpose(np.dot(kWords, self.generatorMatrix)) % 2 # #np.zeros([self.N, self.batchSize], dtype=np.int32) np.repeat(np.expand_dims(testWord, axis=-1), repeats=self.batchSize, axis=1) #
            channelOutput = channelInput
            channelOutput = self.SimulateChannelTF(channelOutput)
            # if(sum(channelOutput) > 0):
            #     waleed = 0
        else:
            channelInput = np.zeros([self.N, self.batchSize], dtype=np.int32)
            channelOutput = channelInput
            if(self.channelType == 'AWGN'): # Voltages need to be after that not binary values
                channelOutput = channelOutput * -2 +1
        return self.CalculateLLRsTF(channelOutput), channelOutput, channelInput