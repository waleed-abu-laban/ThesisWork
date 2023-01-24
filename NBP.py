import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from DataReader import ReadDataTF
from ChannelSimulator import Channel
from NeuralBeliefPropagation import NBPInstance
import LossFunctions as lf
from Initialization import CodeInitialization

tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')

# Parameters ==============================================================================
parityCode = "QHypergraph"
maxFrameErrorCount = 500
maxFrames = 2000000
SaveResults = True
#------------------------------------------------------------------------------------------
TRAINDATA = False
TESTDATA = False
#------------------------------------------------------------------------------------------
iniDescriptor = CodeInitialization(parityCode)
Ns = iniDescriptor.Ns
NKs = iniDescriptor.NKs
lMax = iniDescriptor.lMax
allParameters = iniDescriptor.allParameters
channelType = iniDescriptor.channelType
dataPaths = iniDescriptor.dataPaths
dataPathsOrtho = iniDescriptor.dataPathsOrtho
modelsPaths = iniDescriptor.modelsPaths
resultsPaths = iniDescriptor.resultsPaths
LossFunction = iniDescriptor.LossFunction
initializer = iniDescriptor.initializer
batchSizeTrain = iniDescriptor.batchSizeTrain
learningRate = iniDescriptor.learningRate
epochs = iniDescriptor.epochs
batchSizeTest = iniDescriptor.batchSizeTest

# Simulation ==============================================================================
for counter in range(len(Ns)):
    # inititalize -------------------------------------------------------------------------
    N = Ns[counter]
    NK = NKs[counter]
    parameters = allParameters[counter]
    dataPath = dataPaths[counter]
    modelPath = modelsPaths[counter]
    resultPath = resultsPaths[counter]
    CodeDescriptor = ReadDataTF(dataPath, parityCode)
    IsQuantum = False
    if(parityCode[0] == "Q"):
        IsQuantum = True
        Hortho = np.loadtxt(dataPathsOrtho[counter])
        if(TRAINDATA):
            LossFunction.SetHOrtho(Hortho)
    ChannelObject = Channel(N, batchSizeTrain, channelType, parameters, NK / N, IsQuantum)
    NBPObject = NBPInstance(CodeDescriptor, IsQuantum, lMax, LossFunction, ChannelObject)
    errorRateTotal = []

    # train -------------------------------------------------------------------------
    if(TRAINDATA):
        weights = tf.Variable(initializer(shape=(lMax, CodeDescriptor.EdgesCount)), dtype=tf.float64, trainable=True)
        biases = tf.Variable(initializer(shape=(lMax, N)), dtype=tf.float64, trainable=True)
        NBPObject.SetWeightsBiases(weights, biases)

        losses = tfp.math.minimize(NBPObject.TrainNBP,
                        num_steps=epochs,
                        optimizer=tf.optimizers.Adam(learning_rate=learningRate))
        tf.saved_model.save(NBPObject, modelPath)

    # test -------------------------------------------------------------------------
    if(TESTDATA):
        tensorflow_graph = tf.saved_model.load(modelPath)
        weights = tensorflow_graph.__getattribute__("weights")
        biases = tensorflow_graph.__getattribute__("biases")
        NBPObject.SetWeightsBiases(weights, biases)
        
    # run NBP -------------------------------------------------------------------------
    # inititalize
    NBPObject.SetLossFunctor(lf.LossFunctionNULL())
    NBPObject.ChangeBatchSize(batchSizeTest)
    errorRateTotal = []
    # loop over all parameters ------------------------------------------------------------
    for parameter in parameters:
        # inititalize
        NBPObject.ChangeChannelParameters(np.array([parameter]))
        errorCountTotal = 0
        frameCount = 0
        frameErrorCount = 0
        # check if the processed data is within the test limits
        while((frameErrorCount < maxFrameErrorCount) and (frameCount < maxFrames)):

             # run NBP algorithm
            Edges, decodedWord, channelOutput = NBPObject.NeuralBeliefPropagationOp()
            
            # calculate the error rates ---------------------------------------------------
            # ******************* BLER *******************
            if(IsQuantum):
                decodedWord = decodedWord.numpy()
                frameCount += batchSizeTest

                # Flagged
                originalSynd = NBPObject.CalculateSyndromeError(channelOutput)
                decodedSynd = NBPObject.CalculateSyndromeError(decodedWord)
                syndErrors = np.sum(np.abs(originalSynd - decodedSynd), axis=0)
                filterDataSynd = syndErrors > 0

                # Unflagged
                errorWord = (decodedWord + channelOutput) % 2
                errorWordFiltered = errorWord[:, np.invert(filterDataSynd)]
                logicalOpErrors = np.sum(np.dot(Hortho, errorWord) % 2, axis=0)
                filterDataHmat = logicalOpErrors > 0

                errorCountTotal += np.sum(np.dot(Hortho, errorWord) % 2)
                frameErrorCount += np.sum(1*(filterDataHmat))#(np.sum(1*(filterDataSynd)) + np.sum(1*(filterDataHmat)))
                errorRate = frameErrorCount / frameCount
            # ******************* BER *******************
            else:
                errorCount = np.sum(decodedWord, axis=0)
                errorCountTotal += np.sum(errorCount)
                frameCount += batchSizeTest
                frameErrorCount += np.sum(1*(errorCount > 0))
                errorRate = errorCountTotal / N / frameCount

            # print the data to keep track -------------------------------------------------
            print(N, parameter, frameCount, errorRate, frameErrorCount)
        
        # keep track of the results --------------------------------------------------------
        errorRateTotal.append(errorRate)
        print(errorRateTotal[-1])

    # save the total results from this set of parameters -----------------------------------
    if(SaveResults):
        np.savetxt(resultPath, errorRateTotal)