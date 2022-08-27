import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import ops
from helper_functions import load_code, syndrome
 
def ReadData(path):
    global vNodes, cNodes, vDegrees, cDegrees, weights, biases
    if(parityCode =="LDPC"):
        f = open(path + ".txt", "r")
        NxNK = f.readline()
        NxNK = NxNK.split(" ")
        N = int(NxNK[0])
        NK = int(NxNK[1])
        edgeIndex = 0
        for i in range(N):
            onesPos = f.readline().split(" ")
            if(i == 0):
                vNodesTemp = [[0 for i in range(len(onesPos))] for j in range(N)]
                cNodesTemp = [[] for j in range(NK)]
                #decoder.Edges = tf.zeros(N * len(onesPos), dtype=tf.float64)
            for j in range(len(onesPos)):
                vNodesTemp[i][j] = edgeIndex
                cNodesTemp[int(onesPos[j])].append(edgeIndex)
                edgeIndex += 1
        vNodes = tf.constant(vNodesTemp, dtype=tf.int64)
        cNodes = tf.constant(cNodesTemp, dtype=tf.int64)
        vDegrees = np.repeat(vNodes.shape[1], repeats = vNodes.shape[0])
        cDegrees = np.repeat(cNodes.shape[1], repeats = cNodes.shape[0])
        weights = tf.Variable(initializer(shape=(lMax, N * vNodes.shape[1])), dtype=tf.float64, trainable=True)
        biases = tf.Variable(initializer(shape=(lMax, N)), dtype=tf.float64, trainable=True)
    else:
        code = load_code(path, '')
        H = code.H
        G = code.G
        num_edges = code.num_edges
        cNodesTemp = code.u
        vNodesTemp = code.d
        m = code.m

        N = code.n
        NK = code.k
        vDegrees = code.var_degrees
        cDegrees = code.chk_degrees
        vNodes = tf.ragged.constant(vNodesTemp, dtype=tf.int64)
        cNodes = tf.ragged.constant(cNodesTemp, dtype=tf.int64)
        initializer = tf.keras.initializers.TruncatedNormal(mean = 1.0, stddev=0.0)#, seed=1)
        weights = tf.Variable(initializer(shape=(lMax, num_edges)), dtype=tf.float64, trainable=True)
        biases = tf.Variable(initializer(shape=(lMax, N)), dtype=tf.float64, trainable=True)

# def ReadLLRs(path, N):
#     LLRsTemp = []
#     f = open(path + ".txt", "r")
#     for i in range(N):
#         LLRsTemp.append(float(f.readline()))
#     global LLRs
#     LLRs = tf.constant(LLRsTemp, dtype=tf.float32)

def ReadLLRs(f, N):
    LLRsTemp = np.array([])
    for i in range(N):
        LLRsTemp = np.append(LLRsTemp, float(f.readline()))
    return tf.constant(LLRsTemp, dtype=tf.float64)

def SimulateChannel(y, channelType, parameters = 0, codeRate = 1):
    data = 0 * y
    if(channelType == 'BSC'):
        flipVec = 1*(np.random.rand(y.size) <= parameters) # eps = parameter
        data = 1*np.logical_xor(y, flipVec)
    elif(channelType == 'AWGN'):
        # Symbol mapper
        symbols = y * -2 +1
        # Noise
        EbN0 = 10**(parameters/10) # EbN0dB = parameter
        No = 1/(EbN0 * codeRate)
        noiseVar = No/2
        symbolsShape = symbols.shape
        noise = np.random.randn(symbolsShape[0], symbolsShape[1]) * np.repeat(np.expand_dims(np.sqrt(noiseVar), -1), symbolsShape[1], axis = 1)
        # Modulated data
        data = symbols + noise
    return data

def CalculateLLRs(y, channelType, parameters = 0, codeRate = 1):
    LLRsTemp = y * 0
    if(channelType == 'BSC'):
        y1LLR = np.log(parameters) - np.log(1-parameters) # eps = parameter
        LLRsTemp = (y*2-1) * y1LLR
    elif(channelType == 'AWGN'):
        EbN0 = 10**(parameters/10) # EbN0dB = parameter
        No = 1/(EbN0 * codeRate)
        noiseVar = No/2
        LLRsTemp = y*2 / np.repeat(np.expand_dims(np.sqrt(noiseVar), -1), y.shape[1], axis = 1)
    global LLRs
    LLRs = tf.constant(LLRsTemp, shape=[N, y.size//N], dtype=tf.float64)

def IndexFilter(array, indicesToChange, valuesChange):
    flatValuesChange = tf.reshape(tf.transpose(valuesChange), [-1])
    indexChange = tf.expand_dims(tf.tile(indicesToChange, [LLRs.shape[1]]), -1)
    batchIndices = tf.expand_dims(tf.repeat(tf.range(LLRs.shape[1], dtype=tf.int64), repeats= indicesToChange.shape[0]), -1)
    indicesChange = tf.concat([indexChange, batchIndices], axis = 1)

    #indexChange = tf.stack( [flatIndicesToChange], axis=1 )
    elementsChange = tf.sparse.reorder(tf.SparseTensor(indicesChange, flatValuesChange, tf.shape(array, out_type=tf.int64)))
    flatFilteredArray = tf.sparse.to_dense(elementsChange, default_value = 0.)
    return tf.reshape(flatFilteredArray, tf.shape(array))

def CalculateVMarginals(Edges, iteration):
    weightsIt = tf.tile(tf.reshape(weights[iteration], [-1, 1]), [1, LLRs.shape[1]])
    biasesIt = tf.tile(tf.reshape(biases[iteration], [-1, 1]), [1, LLRs.shape[1]])
    weightedEdges =  tf.gather(Edges, vNodes) * tf.gather(weightsIt, vNodes)
    biasedLLRs = LLRs * biasesIt
    sum = biasedLLRs + tf.reduce_sum(weightedEdges, axis = 1) # Calculate Variable Belief
    flatIndices = tf.reshape(vNodes, [-1])
    newValues =  tf.repeat(sum, repeats=vDegrees, axis = 0) - tf.gather(Edges, flatIndices) * tf.gather(weightsIt, flatIndices) 
    Edges = IndexFilter(Edges, flatIndices, newValues)
    #Edges = tf.clip_by_value(Edges, clip_value_min=-30, clip_value_max=1e200)
    return Edges, sum

def BoxPlusOp(L1, L2):
    maxThreshold = 1e200
    term1 = (1 + tf.keras.activations.exponential(L1 + L2))
    term1 = tf.clip_by_value(term1, -maxThreshold, maxThreshold)
    term2 = (tf.keras.activations.exponential(L1) +tf.keras.activations.exponential(L2))
    term2 = tf.clip_by_value(term2, -maxThreshold, maxThreshold)
    return tf.math.log(term1 / term2)

def BoxPlusDecoder(Edges, index):
    rangeIndices = tf.range(index.shape[0])
    twoDimIndicesF = tf.transpose([rangeIndices, index])
    twoDimIndicesB = tf.transpose([rangeIndices, cDegrees-(1+index)])

    FFlatIndices = tf.gather_nd(cNodes, twoDimIndicesF)
    BFlatIndices = tf.gather_nd(cNodes, twoDimIndicesB)
    if(index.all() == 0):  
        F = tf.expand_dims(tf.gather(Edges, FFlatIndices), -1)
        B = tf.expand_dims(tf.gather(Edges, BFlatIndices), -1)
    else:
        # The recursion is the non regular check nodes case needs more testing
        F, B = BoxPlusDecoder(Edges, np.maximum(index - 1, 0)) 

        result = tf.expand_dims(BoxPlusOp(F[:, :, -1], tf.gather(Edges, FFlatIndices)), -1)
        F = tf.experimental.numpy.append(F, result, axis = -1)

        result = tf.expand_dims(BoxPlusOp(B[:, :, -1], tf.gather(Edges, BFlatIndices)), -1)
        B = tf.experimental.numpy.append(B, result, axis = -1)
    return F, B

def CalculateCMarginals(Edges):
    F, B = BoxPlusDecoder(Edges, cDegrees - 2)
    newEdges = IndexFilter(Edges, tf.gather(cNodes, 0, axis = 1), B[:, :, -1])
    newEdges += IndexFilter(Edges, tf.gather(cNodes, max(cDegrees) - 1, axis = 1), F[:, :, -1])
    for i in np.arange(1, max(cDegrees) - 1, 1):
        # The recursion is the non regular check nodes case needs more testing
        newEdges += IndexFilter(Edges, tf.gather(cNodes, i, axis = 1), BoxPlusOp(F[:, :, i-1], B[:, :, -(i+1)]))
    #newEdges = tf.clip_by_value(newEdges, clip_value_min=-30, clip_value_max=1e200)
    return newEdges

def CalculateSyndrom(Edges):
    return tf.reduce_sum(tf.cast(tf.gather(Edges, cNodes) < 0, tf.int64), axis = 1) % 2

@tf.function
def ContinueCalculation(decodedLLRs, Edges, iteration, loss):
    if(iteration >= lMax):
        return False
    if(tf.reduce_sum(tf.cast(decodedLLRs < 0, tf.int64)) == 0): # zeroCodeWord
        return False
    return tf.reduce_sum(CalculateSyndrom(Edges)) != 0 # isCodeWord other than zero

def BeliefPropagationIt(decodedLLRs, Edges, iteration, loss):
    Edges = CalculateCMarginals(Edges) # Check nodes
    Edges, decodedLLRs = CalculateVMarginals(Edges, iteration) # Variable nodes
    if(TRAINDATA):
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-decodedLLRs, labels=labels)) #/ lMax
        tf.print(loss)
    iteration += 1
    return decodedLLRs, Edges, iteration, loss

def BeliefPropagationOp():
    #Initialize
    Edges = tf.zeros([weights.shape[1], LLRs.shape[1]], dtype=tf.float64)
    Edges = IndexFilter(Edges, tf.reshape(vNodes, [-1]), tf.repeat(LLRs, repeats=vDegrees, axis = 0))
    # parameters: [decodedLLRs, Edges, iteration, loss]
    decodedLLRs, Edges, iteration, loss = \
                        tf.while_loop(\
                         ContinueCalculation,\
                         BeliefPropagationIt,\
                        [tf.ones([N, LLRs.shape[1]], dtype=tf.float64) * -1,\
                         Edges,\
                         tf.constant(0,dtype=tf.int32),\
                         tf.constant(0.0, dtype=tf.float64)])
    if(TRAINDATA):
        return loss
    else:
        return decodedLLRs

def Cost():
    y = np.zeros([batchSize//snrBatchSize, snrBatchSize * N])
    channelOutput = SimulateChannel(y, channelType = channelType, parameters = parameters, codeRate = codeRate)
    CalculateLLRs(channelOutput, channelType = channelType, parameters = parameters, codeRate = codeRate)
    return BeliefPropagationOp()

parityCode = "BCH"
TRAINDATA = False
tf.compat.v1.enable_eager_execution()
# Parameters
if(parityCode == "LDPC"):
    Ns = [1024]#[256, 1024, 2048, 4096, 8192]
    NKs = [ j // 2 for j in Ns]
elif(parityCode == "BCH"):
    Ns = [63]
    NKs = [45]
else: # Polar
    Ns = [128]
    NKs = [64]
channelType = 'AWGN' # BSC or AWGN
lMax = 5
batchSize = 120
snrBatchSize = 20
learningRate = 0.1
epochs = 1
initializer = tf.keras.initializers.TruncatedNormal(mean = 1.0, stddev=0.0)#, seed=1)
tf.keras.backend.set_floatx('float64')
if(parityCode == "LDPC"):
    allParameters = [np.arange(0.5, 3.3, 0.4)] #[np.arange(0.5, 5.1, 0.25), np.arange(0.5, 3.3, 0.4), np.arange(0.5, 2.9, 0.3), np.arange(0.5, 2.3, 0.2), np.arange(0.5, 2.3, 0.2)] 
else:
    allParameters = [np.arange(2,7)]
counter = 0
readLLRs = False
if(readLLRs):
    LLRsPath = "CLLRs"
    f = open(LLRsPath + ".txt", "r")
for N in Ns:
    NK = NKs[counter]
    codeRate = (NK / N)
    parameters = allParameters[counter]
    counter += 1
    if(parityCode == "LDPC"):
        dataPath = "codesLDPC\ldpc_h_reg_dv3_dc6_N" + str(N) + "_Nk" + str(NK)
    elif(parityCode == "BCH"):
        dataPath = "codesHDPC\BCH_" + str(N) + "_" + str(NK) + ".alist"
    else: # Polar
        dataPath = "codesHDPC\polar_" + str(N) + "_" + str(NK) + ".alist"
    maxFrameErrorCount = 500
    maxFrames = 2000000
    # Acquire data (H matrix and LLRs)
    ReadData(dataPath)

    if(TRAINDATA):
        labels = tf.zeros([N, batchSize], dtype=tf.float64)
        losses = tfp.math.minimize(Cost,
                        num_steps=epochs,
                        optimizer=tf.optimizers.Adam(learning_rate=learningRate))

        # for step in range(epochs):
        #     if(readLLRs):
        #         LLRs = ReadLLRs(f, N)
        #     else:
        #         y = np.zeros(N)
        #         channelOutput = SimulateChannel(y, channelType = channelType, parameter = parameter, codeRate = codeRate)
        #         CalculateLLRs(channelOutput, channelType = channelType, parameter = parameter, codeRate = codeRate)
            

        #     optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate).minimize(BeliefPropagationOp, var_list=[weights, biases])
            
        #     if(step % 100 == 0):
        #         print(str(step) + " minibatches completed")
        TRAINDATA = False

    if(not TRAINDATA):
        BER = []
        for parameter in parameters:
            errorCountTotal = 0
            frameCount = 0
            frameErrorCount = 0
            while((frameErrorCount < maxFrameErrorCount) and (frameCount < maxFrames)):
                if(readLLRs):
                    LLRs = ReadLLRs(f, N)
                else:
                    y = np.zeros([N, 1])
                    channelOutput = SimulateChannel(y, channelType = channelType, parameters = np.array([parameter]), codeRate = codeRate)
                    CalculateLLRs(channelOutput, channelType = channelType, parameters = np.array([parameter]), codeRate = codeRate)
                 
                    decodedLLRs = BeliefPropagationOp() # Run BP algorithm
                    
                    errorCount = np.sum(tf.cast(decodedLLRs < 0, tf.int64))
                    errorCountTotal += errorCount
                    frameCount += 1
                    frameErrorCount += (1*(errorCount > 0))
                    #if(frameCount % 100 == 0):
                    print(N, parameter, frameCount, errorCountTotal / N / frameCount, frameErrorCount)
            
            BER.append(errorCountTotal / N / frameCount)
            print(BER[-1])
        np.savetxt(parityCode + "_n"+ str(N) + "_k" + str(NK) + "_BP_Random_" + str(lMax) + "it_AWGN_Fastpy.txt", BER)
        


# # Parameters
# N = 1024
# NK = N//2
# lMax = 10
# EbN0dB = 3
# eps = 0.1
# channelType = 'AWGN' # BSC or AWGN
# codeRate = 1 - (NK / N)
# dataPath = "ldpc_h_reg_dv3_dc6_N" + str(N) + "_Nk" + str(NK)
# readLLRs = False
# LLRsPath = "CLLRs"
# maxFrameErrorCount = 30
# maxFrames = 2000000

# batch_size = 120
# labels = np.zeros([N,batch_size])
# initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev=1.0)#, seed=1)

# # Acquire data (H matrix and LLRs)
# decoder = ReadData(dataPath)
# if(readLLRs):
#     ReadLLRs(LLRsPath, N)
# else:
#     y = np.zeros(N)
#     channelOutput = SimulateChannel(y, channelType = channelType, eps = eps, EbN0dB = EbN0dB, codeRate = codeRate)
#     CalculateLLRs(channelOutput, channelType = channelType, eps = eps, EbN0dB = EbN0dB, codeRate = codeRate)

# decoder.Edges, _ = CalculateVMarginals(decoder.Edges)
# decodedLLRs, decoder.Edges, iteration, loss = BeliefPropagationOp(decoder.Edges)
# print(np.sum(tf.cast(decodedLLRs < 0, tf.int64)))


