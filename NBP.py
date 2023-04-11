import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from DataReader import ReadDataTF
from ChannelSimulator import Channel
from NeuralBeliefPropagation import NBPInstance
import LossFunctions as lf
from Initialization import CodeInitialization
import ResultsCalculator as RC

tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')

# Parameters ==============================================================================
parityCode = "QHypergraph"
maxFrameErrorCount = 500
maxFrames = 10000000
SaveResults = True
isZeroCodeWord = True # Generator matrix should be provided if False
isSyndromeBased = False # Will be True anyway for Quantum case
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
generatorMatrixPaths = iniDescriptor.generatorMatrixPaths

# Simulation ==============================================================================
for counter in range(len(Ns)):
    # inititalize -------------------------------------------------------------------------
    N = Ns[counter]
    NK = NKs[counter]
    parameters = allParameters[counter]
    dataPath = dataPaths[counter]
    modelPath = modelsPaths[counter]
    resultPath = resultsPaths[counter]
    if(isZeroCodeWord):
        generatorMatrix = np.zeros((NK, N))
    else:
        generatorMatrix = np.loadtxt(generatorMatrixPaths[counter])
    CodeDescriptor = ReadDataTF(dataPath, parityCode)
    IsQuantum = False
    if(parityCode[0] == "Q"):
        IsQuantum = True
        isSyndromeBased = True
        Hortho = np.loadtxt(dataPathsOrtho[counter])
        if(TRAINDATA):
            LossFunction.SetHOrtho(Hortho)
        ResultsCalculator = RC.QuantumCalculator(batchSizeTest, maxFrameErrorCount, maxFrames, Hortho)
        #ResultsCalculator = RC.LogicalQuantumCalculator(batchSizeTest, maxFrameErrorCount, maxFrames, np.loadtxt(generatorMatrixPaths[0]), np.loadtxt(generatorMatrixPaths[1]))
    else:
        ResultsCalculator = RC.ClassicalCalculator(batchSizeTest, maxFrameErrorCount, maxFrames) #not passing N means BLER
    ChannelObject = Channel(N, batchSizeTrain, channelType, parameters, NK / N, IsQuantum, generatorMatrix)
    NBPObject = NBPInstance(CodeDescriptor, isSyndromeBased, lMax, LossFunction, ChannelObject)
    errorRateTotal = []

    # train -------------------------------------------------------------------------
    if(TRAINDATA):
        weights = tf.Variable(initializer(shape=(lMax, CodeDescriptor.EdgesCount)), dtype=tf.float64, trainable=True)
        biases =  tf.Variable(tf.ones((lMax, N), dtype=tf.float64), dtype=tf.float64, trainable=True) # tf.Variable(initializer(shape=(lMax, N)), dtype=tf.float64, trainable=True) #

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
        ResultsCalculator.Reset()
        ResultsCalculator.SetParameter(parameter)
        # check if the processed data is within the test limits
        while(ResultsCalculator.Continue()):

            # run NBP algorithm
            Edges, decodedLLRs, channelOutput, channelInput = NBPObject.DecimatedBeliefPropagation(ResultsCalculator) #  NBPObject.NeuralBeliefPropagationOp() #
            # if(np.sum(not NBPObject.terminated) > 0):
            #     decodedWords = NBPObject.ErasurePropagationOp(decodedLLRs)
            # else:
            decodedWords = tf.cast(decodedLLRs < 0, tf.int64)

            # calculate the error rates ---------------------------------------------------
            if(isSyndromeBased):
                errorWord = (decodedWords.numpy() + channelOutput + channelInput) % 2
            else:
                errorWord = (decodedWords.numpy() + channelInput) % 2
            # ******************* Quantum *******************
            if(IsQuantum):
                errorRate = ResultsCalculator.CaclulateErrorRate(errorWord, NBPObject.terminated)
            # ******************* Classical *******************
            else:
                errorRate = ResultsCalculator.CaclulateErrorRate(errorWord)
        
        # keep track of the results --------------------------------------------------------
        errorRateTotal.append(errorRate)
        print(errorRateTotal[-1])

    # save the total results from this set of parameters -----------------------------------
    if(SaveResults):
        np.savetxt(resultPath, errorRateTotal)