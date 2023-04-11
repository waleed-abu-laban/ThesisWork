import LossFunctions as lf
import tensorflow as tf
import numpy as np

class iniDescriptor:
	def __init__(self):
            None
def CodeInitialization(parityCode):
    generatorMatrixPaths = []
    dataPathsOrtho = []
    if(parityCode == "LDPC"):
        Ns = [1024]#[256, 1024, 2048, 4096, 8192]
        NKs = [ j // 2 for j in Ns]
        lMax = 50
        allParameters = [np.array([0.085])]#[np.arange(0.5, 5.1, 0.25), np.arange(0.5, 3.3, 0.4), np.arange(0.5, 2.9, 0.3), np.arange(0.5, 2.3, 0.2), np.arange(0.5, 2.3, 0.2)]
        channelType = 'BSC' # BSC or AWGN
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
        Ns = [63]#[15]
        NKs = [45]#[11]
        lMax = 5
        allParameters = [np.arange(1,7)] # [np.array([0.001])] #
        channelType = 'AWGN' # BSC or AWGN
        dataPaths = []
        modelsPaths = []
        resultsPaths = []
        generatorMatrixPaths = []
        for i in range(len(Ns)):
            N = Ns[i]
            NK = NKs[i]
            dataPaths.append("codesHDPC\BCH_" + str(N) + "_" + str(NK) + ".alist")
            modelsPaths.append("Models\\" + str(parityCode) + "_N" + str(N) + "_Nk" + str(NK))
            resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
            generatorMatrixPaths.append("codesHDPC\GBCH_" + str(N) + "_" + str(NK) + ".txt")
        LossFunction = lf.LossFunctionCE()
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.7, stddev=0.5, seed=1)
        batchSizeTrain = 120
        learningRate = 0.001
        epochs = 5001
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
    elif(parityCode == "QShor"):
        Ns = [9 * 2] #[129]
        NKs = [1]
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
            dataPaths.append("codesQLDPC\H_Shor.alist")
            dataPathsOrtho.append("codesQLDPC\HOrthoM_Shor.txt")
            modelsPaths.append("Models\\" + str(parityCode) + "_N" + str(N//2) + "_Nk" + str(NK))
            resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N//2) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
        
        LossFunction = lf.LossFunctionLiu()
        initializer = tf.keras.initializers.TruncatedNormal(mean=1.0, stddev=0.0, seed=1)
        batchSizeTrain = 120
        learningRate = 0.0001
        epochs = 10001
        batchSizeTest = 1
    #------------------------------------------------------------------------------------------
    elif(parityCode == "QBicycle"):
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
    elif(parityCode == "QHypergraph"):
        Ns = [129] # [129 * 2] # [400] #
        NKs = [28] # [16]# [16] #
        lMax = 100
        allParameters = [np.array([1e-2])] #[np.array([1e-2, 8e-3, 6e-3, 4e-3, 2e-3, 1e-3, 8e-4, 6e-4])] #[np.linspace(0.01, 0.05, 6)] #[np.array([1e-2, 8e-3, 6e-3, 4e-3, 2e-3, 1e-3, 8e-4, 6e-4, 4e-4])] #[np.array([1e-2, 5.9e-3, 3.5e-3, 2.1e-3, 1.2e-3, 7.8e-4, 4.5e-4, 2.7e-4, 1.6e-4, 9.7e-5])] #[np.arange(1e-2, 5e-2, 7e-3)]
        channelType = 'BSC'
        dataPaths = []
        dataPathsOrtho = []
        modelsPaths = []
        resultsPaths = []
        for i in range(len(Ns)):
            N = Ns[i]
            NK = NKs[i]
            dataPaths.append("codesQLDPC\Hz.alist")#  ("codesQLDPC\Hz_400_16.alist")# ("codesQLDPC\Hypergraph_" + str(N//2) + "_" + str(NK) + ".alist") #("codesQLDPC\H_new.alist") 
            dataPathsOrtho.append("codesQLDPC\HzOrtho.txt")#   ("codesQLDPC\HzOrtho_400_16.txt")# ("codesQLDPC\HorthoMatrix_Hypergraph_" + str(N//2) + "_" + str(NK) + ".txt") #("codesQLDPC\Hortho_new.txt") #("codesQLDPC\Hz.txt") ("codesQLDPC\logicalWords.txt")# 
            modelsPaths.append("Models\\QHypergraph_N129_Nk28_Liu10^4")#("Models\\" + str(parityCode) + "_N" + str(N//2) + "_Nk" + str(NK))
            resultsPaths.append("Results\\" + str(parityCode) + "_N"+ str(N//2) + "_NK" + str(NK) + "_NBP_" + str(lMax) + "it_" + str(channelType) + ".txt")
        
        generatorMatrixPaths.append("codesQLDPC\Lx.txt")
        generatorMatrixPaths.append("codesQLDPC\Lz.txt")

        LossFunction = lf.LossFunctionLiu() #  LossFunctionLiu
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.7, stddev=0.5, seed=1) #(mean=0.0, stddev=0.5, seed=1)
        batchSizeTrain = allParameters[0].size * 20
        learningRate = 0.001
        epochs = 1000
        batchSizeTest = 100
    #------------------------------------------------------------------------------------------

    iniDescriptor.Ns = Ns
    iniDescriptor.NKs = NKs
    iniDescriptor.lMax = lMax
    iniDescriptor.allParameters = allParameters
    iniDescriptor.channelType = channelType
    iniDescriptor.dataPaths = dataPaths
    iniDescriptor.dataPathsOrtho = dataPathsOrtho
    iniDescriptor.modelsPaths = modelsPaths
    iniDescriptor.resultsPaths = resultsPaths
    iniDescriptor.LossFunction = LossFunction
    iniDescriptor.initializer = initializer
    iniDescriptor.batchSizeTrain = batchSizeTrain
    iniDescriptor.learningRate = learningRate
    iniDescriptor.epochs = epochs
    iniDescriptor.batchSizeTest = batchSizeTest
    iniDescriptor.generatorMatrixPaths = generatorMatrixPaths
    return iniDescriptor
        