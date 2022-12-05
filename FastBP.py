import numpy as np
from BeliefPropagation import BPInstance
from DataReader import ReadData
from ChannelSimulator import SimulateChannel, CalculateLLRs

# Parameters ==============================================================================
parityCode = "QHypergraph"
maxFrameErrorCount = 500
maxFrames = 200000
SaveData = True
#------------------------------------------------------------------------------------------
if(parityCode == "LDPC"):
    Ns = [256, 1024, 2048, 4096, 8192]
    NKs = [ j // 2 for j in Ns]
    lMax = 12
    allParameters = [np.arange(0.5, 5.1, 0.25), np.arange(0.5, 3.3, 0.4), np.arange(0.5, 2.9, 0.3), np.arange(0.5, 2.3, 0.2), np.arange(0.5, 2.3, 0.2)]
    channelType = 'AWGN' # BSC or AWGN
    dataPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesLDPC\ldpc_h_reg_dv3_dc6_N" + str(N) + "_Nk" + str(NK))
#------------------------------------------------------------------------------------------
elif(parityCode == "BCH"):
    Ns = [63]
    NKs = [45]
    lMax = 5
    allParameters = [np.array([6])]#[np.arange(1,7)]
    channelType = 'AWGN'
    dataPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesHDPC\BCH_" + str(N) + "_" + str(NK) + ".alist")
#------------------------------------------------------------------------------------------
elif(parityCode == "Polar"):
    Ns = [128]
    NKs = [64]
    lMax = 12
    allParameters = [np.arange(1,7)]
    channelType = 'AWGN'
    dataPaths = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesHDPC\polar_" + str(N) + "_" + str(NK) + ".alist")
#------------------------------------------------------------------------------------------
elif(parityCode == "QHypergraph"):
    Ns = [129 * 2]
    NKs = [28]
    lMax = 12
    allParameters = [[1e-2, 5.9e-3, 3.5e-3, 2.1e-3, 1.2e-3, 7.8e-4, 4.5e-4, 2.7e-4, 1.6e-4, 9.7e-5]]
    channelType = 'BSC'
    dataPaths = []
    dataPathsOrtho = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesQLDPC\Hypergraph_" + str(N//2) + "_" + str(NK) + ".alist")
        dataPathsOrtho.append("codesQLDPC\HorthoMatrix_Hypergraph_" + str(N//2) + "_" + str(NK) + ".txt")
#------------------------------------------------------------------------------------------
else: # QBicycle
    Ns = [256 * 2]
    NKs = [32]
    lMax = 12
    allParameters = [[1e-2, 3e-3, 1e-3, 3e-4, 1e-4]]
    channelType = 'BSC'
    dataPaths = []
    dataPathsOrtho = []
    for i in range(len(Ns)):
        N = Ns[i]
        NK = NKs[i]
        dataPaths.append("codesQLDPC\Bicycle_" + str(N//2) + "_" + str(NK) + ".alist")
        dataPathsOrtho.append("codesQLDPC\HorthoMatrix_Bicycle_" + str(N//2) + "_" + str(NK) + ".txt")
#------------------------------------------------------------------------------------------

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