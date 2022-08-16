import numpy as np

def ReadData(path):
    f = open(path + ".txt", "r")
    NxNK = f.readline()
    NxNK = NxNK.split(" ")
    N = int(NxNK[0])
    NK = int(NxNK[1])
    global vNodes
    global cNodes
    global Edges
    edgeIndex = 0
    for i in range(N):
        onesPos = f.readline().split(" ")
        if(i == 0):
            vNodes = np.zeros([N, len(onesPos)], dtype=int) #[[0 for i in range(len(onesPos))] for j in range(N)]
            cNodesTemp = [[] for j in range(NK)]
            Edges = np.zeros(N * len(onesPos))
        for j in range(len(onesPos)):
            vNodes[i][j] = edgeIndex #int(onesPos[j])
            cNodesTemp[int(onesPos[j])].append(edgeIndex)
            edgeIndex += 1
    cNodes = np.array(cNodesTemp, dtype=int)

def ReadLLRs(f, N):
    LLRs = np.array([])
    for i in range(N):
        LLRs = np.append(LLRs, float(f.readline()))
    return LLRs

def SimulateChannel(y, channelType, parameter = 0, codeRate = 1):
    data = 0 * y
    if(channelType == 'BSC'):
        flipVec = 1*(np.random.rand(y.size) <= parameter) # eps = parameter
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
    global LLRs
    LLRs = y * 0
    if(channelType == 'BSC'):
        y1LLR = np.log(parameter) - np.log(1-parameter) # eps = parameter
        LLRs = (y*2-1) * y1LLR
    elif(channelType == 'AWGN'):
        EbN0 = 10**(parameter/10) # EbN0dB = parameter
        No = 1/(EbN0 * codeRate)
        noiseVar = No/2
        LLRs = y*2 / noiseVar

def ThresholdClip():
    threshold = 30
    Edges[Edges < -threshold] = -threshold
    #Edges[Edges > threshold] = threshold # Why not there?

def CalculateVMarginals():
    sum = LLRs + np.sum(Edges[np.array(vNodes)], axis = 1) # Calculate Variable Belief
    Edges[np.array(vNodes).flatten()] = np.repeat(sum, vNodes.shape[1]) - Edges[np.array(vNodes).flatten()]
    ThresholdClip()
    return sum

def BoxPlusOp(L1, L2):
    maxThreshold = 1e30 #1e200
    term1 =(1 + np.exp(L1 + L2))
    term1[term1 < -maxThreshold] = -maxThreshold
    term1[term1 > maxThreshold] = maxThreshold
    term2 = (np.exp(L1) + np.exp(L2))
    term2[term2 < -maxThreshold] = -maxThreshold
    term2[term2 > maxThreshold] = maxThreshold
    return np.log(term1 / term2)

def BoxPlusDecoder(index):
    if(index == 0):
        F = np.expand_dims(Edges[np.array(cNodes[:, index]).flatten()], axis = 1)

        B = np.expand_dims(Edges[np.array(cNodes[:, -(1+index)]).flatten()], axis = 1)
    else:
        F, B = BoxPlusDecoder(index - 1)

        result = np.expand_dims(BoxPlusOp(F[:, -1], Edges[np.array(cNodes[:, index]).flatten()]), axis = 1)
        F = np.append(F, result, axis = 1)

        result = np.expand_dims(BoxPlusOp(B[:, -1], Edges[np.array(cNodes[:, -(1+index)]).flatten()]), axis = 1)
        B = np.append(B, result, axis = 1)
    return F, B

def CalculateCMarginals():
    F, B = BoxPlusDecoder(cNodes.shape[1] - 2)
    Edges[np.array(cNodes[:, 0]).flatten()] = B[:, -1]
    Edges[np.array(cNodes[:, -1]).flatten()] = F[:, -1]
    for i in np.arange(1, cNodes.shape[1] - 1, 1):
        Edges[np.array(cNodes[:, i]).flatten()] = BoxPlusOp(F[:, i-1], B[:, -(i+1)])
    #Edges[np.array(cNodes).flatten()] *= np.repeat(syndrom, cNodes.shape[1])
    ThresholdClip()

def CalculateSyndrom():
    return np.sum(1 * (Edges[np.array(cNodes)] < 0), axis = 1) % 2

def CodeWordTermination(yLLR):
    if(np.sum( 1*(yLLR < 0)) == 0): # zeroCodeWord
        return True
    return np.sum(CalculateSyndrom()) == 0 # isCodeWord other than zero

def BeliefPropagation(lMax):
    CalculateVMarginals() # Initialize
    #global syndrom
    #syndrom = np.power(-1, CalculateSyndrom())
    for i in range(lMax):
        CalculateCMarginals() # Check nodes
        decodedLLRs = CalculateVMarginals() # Variable nodes
        if(CodeWordTermination(decodedLLRs)):
            break
    decodedWord = 1*(decodedLLRs < 0)
    return decodedLLRs, decodedWord

# Parameters
Ns = [256, 1024, 2048, 4096, 8192]
channelType = 'AWGN' # BSC or AWGN
lMax = 10
allParameters = [np.arange(0.5, 5.1, 0.25), np.arange(0.5, 3.3, 0.4), np.arange(0.5, 2.9, 0.3), np.arange(0.5, 2.3, 0.2), np.arange(0.5, 2.3, 0.2)] 
counter = 0
readLLRs = False
LLRs = np.array([])
for N in Ns:
    NK = N//2
    codeRate = 1 - (NK / N)
    parameters = allParameters[counter]
    counter += 1
    dataPath = "ldpc_h_reg_dv3_dc6_N" + str(N) + "_Nk" + str(NK)
    maxFrameErrorCount = 500
    maxFrames = 2000000
    # Acquire data (H matrix and LLRs)
    ReadData(dataPath)
    if(readLLRs):
        LLRsPath = "CLLRs"
        f = open(LLRsPath + ".txt", "r")

    BER = []
    for parameter in parameters:
        errorCountTotal = 0
        frameCount = 0
        frameErrorCount = 0
        while((frameErrorCount < maxFrameErrorCount) and (frameCount < maxFrames)):
            if(readLLRs):
                LLRs = ReadLLRs(f, N)
            else:
                y = np.zeros(N)
                channelOutput = SimulateChannel(y, channelType = channelType, parameter = parameter, codeRate = codeRate)
                CalculateLLRs(channelOutput, channelType = channelType, parameter = parameter, codeRate = codeRate)
            
            Edges *= 0
            decodedLLRs, decodedWord = BeliefPropagation(lMax) # Run BP algorithm
            errorCount = np.sum(decodedWord)
            errorCountTotal += errorCount
            frameCount += 1
            frameErrorCount += (1*(errorCount > 0))
            if(frameCount % 100 == 0):
                print(N, parameter, frameCount, errorCountTotal / N / frameCount, frameErrorCount)
            
        BER.append(errorCountTotal / N / frameCount)
        print(BER[-1])
    np.savetxt("LDPC_(3,6)_n"+ str(N) + "_k" + str(NK) + "_BP_Random_" + str(lMax) + "it_AWGN_Fastpy.txt", BER)
    if(readLLRs):
        f.close()