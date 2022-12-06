import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from DataReader import ReadDataTF
from ChannelSimulator import Channel
from NeuralBeliefPropagation import NBPInstance
import LossFunctions as lf

tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')

# Parameters ==============================================================================
parityCode = "QHypergraph"
maxFrameErrorCount = 500
maxFrames = 200000
SaveResults = True
#------------------------------------------------------------------------------------------
TRAINDATA = False
TESTDATA = False
#------------------------------------------------------------------------------------------
if(parityCode == "LDPC"):
    Ns = [256, 1024, 2048, 4096, 8192]
    NKs = [ j // 2 for j in Ns]
    lMax = 50
    allParameters = [np.arange(0.5, 5.1, 0.25), np.arange(0.5, 3.3, 0.4), np.arange(0.5, 2.9, 0.3), np.arange(0.5, 2.3, 0.2), np.arange(0.5, 2.3, 0.2)]
    channelType = 'AWGN' # BSC or AWGN
    dataPaths = []
    modelsPaths = []
    resultsPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesLDPC\ldpc_h_reg_dv3_dc6_N" + str(N) + "_Nk" + str(NK))
        modelsPaths.append("Models\\" + str(parityCode) + "_N" + str(N) + "_Nk" + str(NK))
        resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
    
    LossFunction = lf.LossFunctionCE()
    initializer = tf.keras.initializers.TruncatedNormal(mean=1.0, stddev=0.0, seed=1)
    batchSizeTrain = 120
    learningRate = 0.0001
    epochs = 10001
    batchSizeTest = 100
#------------------------------------------------------------------------------------------
elif(parityCode == "BCH"):
    Ns = [63]
    NKs = [45]
    lMax = 5
    allParameters = [np.arange(1,7)]
    channelType = 'AWGN'
    dataPaths = []
    modelsPaths = []
    resultsPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesHDPC\BCH_" + str(N) + "_" + str(NK) + ".alist")
        modelsPaths.append("Models\\" + str(parityCode) + "_N" + str(N) + "_Nk" + str(NK))
        resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
    
    LossFunction = lf.LossFunctionCE()
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.7, stddev=0.5, seed=1)
    batchSizeTrain = 120
    learningRate = 0.001
    epochs = 10001
    batchSizeTest = 1000
#------------------------------------------------------------------------------------------
elif(parityCode == "Polar"):
    Ns = [128]
    NKs = [64]
    lMax = 12
    allParameters = [np.arange(1,7)]
    channelType = 'AWGN'
    dataPaths = []
    modelsPaths = []
    resultsPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesHDPC\polar_" + str(N) + "_" + str(NK) + ".alist")
        modelsPaths.append("Models\\" + str(parityCode) + "_N" + str(N) + "_Nk" + str(NK))
        resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
    
    LossFunction = lf.LossFunctionCE()
    initializer = tf.keras.initializers.TruncatedNormal(mean=1.0, stddev=0.0, seed=1)
    batchSizeTrain = 120
    learningRate = 0.0001
    epochs = 10001
    batchSizeTest = 1000
#------------------------------------------------------------------------------------------
elif(parityCode == "QHypergraph"):
    Ns = [129 * 2]
    NKs = [28]
    lMax = 12
    allParameters = [np.array([1e-2, 5.9e-3, 3.5e-3, 2.1e-3, 1.2e-3, 7.8e-4, 4.5e-4, 2.7e-4, 1.6e-4, 9.7e-5])] #[np.arange(1e-2, 5e-2, 7e-3)]
    channelType = 'BSC'
    dataPaths = []
    dataPathsOrtho = []
    modelsPaths = []
    resultsPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesQLDPC\Hypergraph_" + str(N//2) + "_" + str(NK) + ".alist")
        dataPathsOrtho.append("codesQLDPC\HorthoMatrix_Hypergraph_" + str(N//2) + "_" + str(NK) + ".txt")
        modelsPaths.append("Models\\" + str(parityCode) + "_N" + str(N//2) + "_Nk" + str(NK))
        resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N//2) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
    
    LossFunction = lf.LossFunctionLiu()
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.7, stddev=0.5, seed=1)
    batchSizeTrain = 120
    learningRate = 0.0001
    epochs = 10001
    batchSizeTest = 100
#------------------------------------------------------------------------------------------
else: # QBicycle
    Ns = [256 * 2]
    NKs = [32]
    lMax = 12
    allParameters = [np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])]
    channelType = 'BSC'
    dataPaths = []
    dataPathsOrtho = []
    modelsPaths = []
    resultsPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesQLDPC\Bicycle_" + str(N//2) + "_" + str(NK) + ".alist")
        dataPathsOrtho.append("codesQLDPC\HorthoMatrix_Bicycle_" + str(N//2) + "_" + str(NK) + ".txt")
        modelsPaths.append("Models\\" + str(parityCode) + "_N" + str(N//2) + "_Nk" + str(NK))
        resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N//2) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
    
    LossFunction = lf.LossFunctionLiu()
    initializer = tf.keras.initializers.TruncatedNormal(mean=1.0, stddev=0.0, seed=1)
    batchSizeTrain = 120
    learningRate = 0.0001
    epochs = 10001
    batchSizeTest = 1000
#------------------------------------------------------------------------------------------

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
    ChannelObject = Channel(N, batchSizeTrain, channelType, parameters, NK / N)
    IsQuantum = False
    if(parityCode[0] == "Q"):
        IsQuantum = True
        Hortho = np.loadtxt(dataPathsOrtho[counter])
        if(TRAINDATA):
            LossFunction.SetHOrtho(Hortho)
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
                filterData = np.sum(NBPObject.CalculateSyndrome(Edges), axis = 0) > 0
                filteredDecodedWord = decodedWord.numpy()[:, filterData]
                filteredChannelOutput = channelOutput[:, filterData]

                errorTotal = (filteredDecodedWord + filteredChannelOutput) % 2
                errorCount = np.sum(np.dot(Hortho, errorTotal) % 2, axis = 0)
                frameErrorCount += np.sum(1*(errorCount > 0))
                frameCount += batchSizeTest
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