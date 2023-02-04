import numpy as np
from BeliefPropagation import BPInstance
from DataReader import ReadData
from ChannelSimulator import SimulateChannel, CalculateLLRs
from Initialization import CodeInitialization

# Parameters ==============================================================================
parityCode = "Hypergraph"
maxFrameErrorCount = 500
maxFrames = 200000
SaveData = True
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
    BPObject = BPInstance(ReadData(dataPath, parityCode))
    if(parityCode[0] == "Q"):
        Hortho = np.loadtxt(dataPathsOrtho[counter])
    errorRateTotal = []

    saveLLRs = []
    isFirst = True

    # loop over all parameters ------------------------------------------------------------
    for parameter in parameters:
        # inititalize ---------------------------------------------------------------------
        errorCountTotal = 0
        frameCount = 0
        frameErrorCount = 0
        # check if the processed data is within the test limits
        while((frameErrorCount < maxFrameErrorCount) and (frameCount < maxFrames)):
            # simulate the all zero codeword ----------------------------------------------
            y = np.zeros(N)
            channelOutput = SimulateChannel(y, channelType = channelType, parameter = parameter, codeRate = NK / N)
            LLRs = CalculateLLRs(channelOutput, channelType = channelType, parameter = parameter, codeRate = NK / N)

            saveLLRs.append(LLRs)
            np.savetxt("LLRsTest.txt", np.transpose(np.array(LLRs)))

            # initialize the BP edges ----------------------------------------------------
            BPObject.ClearEdges()

            # read the syndrom as input ---------------------------------------------------
            syndrome = -1
            if(parityCode[0] == "Q"):
                BPObject.SetLLRs(LLRs)
                BPObject.CalculateVMarginals()
                syndrome = np.power(-1, BPObject.CalculateSyndrome())
                LLRs = CalculateLLRs(y, channelType = channelType, parameter = parameter)
                BPObject.ClearEdges()

            # set the data processed from the parameters ----------------------------------
            BPObject.SetLLRs(LLRs)
            BPObject.SetSyndrome(syndrome)
            decodedLLRs, decodedWord, resultsC, resultsV = BPObject.BeliefPropagation(lMax) # Run BP algorithm

            # calculate the error rates ---------------------------------------------------
            # ******************* BLER *******************
            if(parityCode[0] == "Q"):
                if(not BPObject.IsCodeWord()):
                    errorTotal = (decodedWord + channelOutput) % 2
                    errorCount = np.sum(np.dot(Hortho, errorTotal) % 2)
                    frameErrorCount += (1*(errorCount > 0))
                frameCount += 1
                errorRate = frameErrorCount / frameCount
            # ******************* BER *******************
            else: 
                errorCount = np.sum(decodedWord)
                errorCountTotal += errorCount
                frameErrorCount += (1*(errorCount > 0))
                frameCount += 1
                errorRate = errorCountTotal / N / frameCount
                errorRate = frameErrorCount / frameCount

            # print the data to keep track -------------------------------------------------
            if(frameCount % 100 == 0):
                print(N, parameter, frameCount, errorRate, frameErrorCount)

        # keep track of the results --------------------------------------------------------
        errorRateTotal.append(errorRate)
        print(errorRateTotal[-1])

    # save the total results from this set of parameters -----------------------------------
    if(SaveData):
        if(parityCode[0] == "Q"):
            N = N//2
        np.savetxt(parityCode + "_n"+ str(N) + "_k" + str(NK) + "_BP_Random_" + str(lMax) + "it_" + channelType + "_Fastpy.txt", errorRateTotal)
    
    np.savetxt("LLRsTest.txt", np.transpose(np.array(saveLLRs)))
    np.savetxt("resultsTest.txt", np.transpose(np.array(resultsC)))