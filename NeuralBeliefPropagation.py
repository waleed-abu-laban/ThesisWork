import numpy as np
import tensorflow as tf
from MathFunctions import Fermi, IsFloat
from ldpc.mod2 import rank
ERASURE = 0.3333

class NBPInstance(tf.Module):
    def __init__(self, CodeDescriptor, isSyndromeBased, lMax, LossFunctor, Channel):
        super().__init__()
        self.EdgesCount = CodeDescriptor.EdgesCount
        self.vNodes = CodeDescriptor.vNodes
        self.cNodes = CodeDescriptor.cNodes
        self.vDegrees = CodeDescriptor.vDegrees
        self.cDegrees = CodeDescriptor.cDegrees
        self.isSyndromeBased = isSyndromeBased
        self.lMax = lMax
        self.LossFunction = LossFunctor
        self.Channel = Channel
        self.SetSyndrome(tf.zeros((self.vNodes.shape[0], Channel.GetBatchSize())))
        self.weights = tf.ones(shape=(lMax, CodeDescriptor.EdgesCount), dtype=tf.float64)
        self.biases = tf.ones(shape=(lMax, CodeDescriptor.vNodes.shape[0]), dtype=tf.float64)
        self.H = tf.constant(np.loadtxt("codesQLDPC\Hz.txt"), dtype=tf.int32)#(np.loadtxt("codesQLDPC\HMatrix_Hypergraph_129_28.txt"), dtype=tf.int32) #  (np.loadtxt("codesQLDPC\Hz_400_16.txt"), dtype=tf.int32) #
        self.Hx = tf.constant(np.loadtxt("codesQLDPC\Hx.txt"), dtype=tf.int32)#  (np.loadtxt("codesQLDPC\Hx_400_16.txt"), dtype=tf.int32) #
        tf.random.set_seed(100)

    def SetLossFunctor(self, LossFunctor):
        self.LossFunction = LossFunctor

    def FillEdges(self, Edges, values):
        return self.IndexFilter(Edges, tf.reshape(self.vNodes, [-1]), tf.repeat(values, repeats=self.vDegrees, axis = 0))

    def SetLLRs(self, LLRs):
        self.LLRs = LLRs
        self.batchSize = LLRs.shape[1]

    def SetSyndrome(self, LLRs):
        self.batchSize = LLRs.shape[1]
        self.syndrome = tf.pow(tf.constant(-1, dtype=tf.int64), self.CalculateSyndromeLLRs(LLRs))
       
    def SetWeightsBiases(self, weights, biases):
        self.weights = weights
        self.biases = biases
       
    def GetLossFunction(self):
        return self.LossFunction

    def ChangeBatchSize(self, newBatchSize):
        self.Channel.SetBatchSize(newBatchSize)
        self.batchSize = newBatchSize
        self.SetSyndrome(tf.zeros([self.vNodes.shape[0], newBatchSize]))

    def ChangeChannelParameters(self, newChannelParameters):
        self.Channel.SetChannelParameters(newChannelParameters)

    def IndexFilter(self, array, indicesToChange, valuesChange):
        flatValuesChange = tf.reshape(tf.transpose(valuesChange), [-1])
        indexChange = tf.expand_dims(tf.tile(indicesToChange, [array.shape[1]]), -1)
        batchIndices = tf.expand_dims(tf.repeat(tf.range(array.shape[1], dtype=tf.int64), repeats= indicesToChange.shape[0]), -1)
        indicesChange = tf.concat([indexChange, batchIndices], axis = 1)

        elementsChange = tf.sparse.reorder(tf.SparseTensor(indicesChange, flatValuesChange, tf.shape(array, out_type=tf.int64)))
        flatFilteredArray = tf.sparse.to_dense(elementsChange, default_value = 0)
        return tf.reshape(flatFilteredArray, tf.shape(array))

    def CalculateVMarginals(self, Edges, iteration, LLRs):
        weightsIt = tf.tile(tf.reshape(self.weights[iteration], [-1, 1]), [1, Edges.shape[1]])
        biasesIt = tf.tile(tf.reshape(self.biases[iteration], [-1, 1]), [1, Edges.shape[1]])
        weightedEdges =  tf.gather(Edges, self.vNodes) * tf.gather(weightsIt, self.vNodes)
        biasedLLRs = LLRs * biasesIt
        weightedEdgesTest = weightedEdges.numpy()
        sum = biasedLLRs + tf.reduce_sum(weightedEdges, axis = 1) # Calculate Variable Belief
        flatIndices = tf.reshape(self.vNodes, [-1])
        newValues =  tf.repeat(sum, repeats=self.vDegrees, axis = 0) - tf.gather(Edges, flatIndices) * tf.gather(weightsIt, flatIndices) 
        Edges = self.IndexFilter(Edges, flatIndices, newValues)

        Edges = tf.clip_by_value(Edges, clip_value_min=-20, clip_value_max=20)
        return Edges, sum
    
    def CalculateVErasures(self, Edges, codeWords):
        orderedEdges =  tf.gather(Edges, self.vNodes)
        isErasure, isNotErasure = IsFloat(codeWords)
        filterNewCodeWord = codeWords * isNotErasure + isErasure * tf.reduce_sum(orderedEdges, axis = 1) #This is wrong, if one is not erasure it is enough
        
        orderedEdgesTest = orderedEdges.numpy()
        isErasureTest = isErasure.numpy()
        filterNewCodeWordTest = filterNewCodeWord.numpy()

        flatIndices = tf.reshape(self.vNodes, [-1])
        newValues =  tf.repeat(filterNewCodeWord, repeats=self.vDegrees, axis = 0) - (tf.gather(Edges, flatIndices) * tf.repeat(isErasure, repeats=self.vDegrees, axis = 0)) 
        
        fullErasure = ERASURE * tf.ones_like(orderedEdges)
        fullErasureCompare = tf.reduce_sum(fullErasure, axis = 1)
        fullErasureCompareEdges = tf.repeat(fullErasureCompare, repeats=self.vDegrees, axis = 0) - ERASURE
        isErasureLogical = tf.abs(fullErasureCompareEdges - newValues) < 0.001
        isNotErasureLogical = tf.logical_not(isErasureLogical)

        newValues = newValues * tf.repeat(isNotErasure, repeats=self.vDegrees, axis = 0) + \
        tf.repeat(isErasure, repeats=self.vDegrees, axis = 0) * (tf.cast(newValues > fullErasureCompareEdges, tf.float64) * tf.cast(isNotErasureLogical, tf.float64) + tf.cast(isErasureLogical, tf.float64) * ERASURE)
        Edges = self.IndexFilter(Edges, flatIndices, newValues)

        isErasureLogical = tf.abs(fullErasureCompare - filterNewCodeWord) < 1e-9
        isNotErasureLogical = tf.logical_not(isErasureLogical)
        filterNewCodeWord = filterNewCodeWord * isNotErasure + \
        isErasure * (tf.cast(filterNewCodeWord > fullErasureCompare, tf.float64) * tf.cast(isNotErasureLogical, tf.float64) + tf.cast(isErasureLogical, tf.float64) * ERASURE)

        fullErasureCompareTest = fullErasureCompare.numpy()
        isErasureLogicalTest = isErasureLogical.numpy()
        filterNewCodeWordTest2 = filterNewCodeWord.numpy()

        return Edges, filterNewCodeWord

    def CalculateCErasures(self, Edges, Syndrome):
        orderedEdges =  tf.gather(Edges, self.cNodes)
        sum = tf.reduce_sum(orderedEdges, axis = 1) + Syndrome
        
        flatIndices = tf.reshape(self.cNodes, [-1])
        newValues =  tf.repeat(sum, repeats=self.cDegrees, axis = 0) - tf.gather(Edges, flatIndices) 
        Edges = self.IndexFilter(Edges, flatIndices, newValues)

        isErasure, isNotErasure = IsFloat(Edges)
        Edges = Edges % 2 * isNotErasure + isErasure * ERASURE

        return Edges

    def BoxPlusOp(self, L1, L2):
        maxThreshold = 1e70
        term1 = (1 + tf.keras.activations.exponential(L1 + L2))
        term1 = tf.clip_by_value(term1, -maxThreshold, maxThreshold)
        term2 = (tf.keras.activations.exponential(L1) +tf.keras.activations.exponential(L2))
        term2 = tf.clip_by_value(term2, -maxThreshold, maxThreshold)
        return tf.math.log(term1 / term2)

    def BoxPlusDecoder(self, Edges, index):
        rangeIndices = tf.range(self.cDegrees.shape[0])
        mask1 = tf.minimum(self.cDegrees-1, index)
        mask2 = tf.minimum(self.cDegrees-1, max(self.cDegrees)-(1+index))
        twoDimIndicesF = tf.transpose([rangeIndices, mask1])
        twoDimIndicesB = tf.transpose([rangeIndices, mask2])

        FFlatIndices = tf.gather_nd(self.cNodes, twoDimIndicesF)
        BFlatIndices = tf.gather_nd(self.cNodes, twoDimIndicesB)
        if(index == 0):  
            F = tf.expand_dims(tf.gather(Edges, FFlatIndices), -1)
            B = tf.expand_dims(tf.gather(Edges, BFlatIndices), -1)
        else:
            F, B = self.BoxPlusDecoder(Edges, index - 1) 

            result = tf.expand_dims(self.BoxPlusOp(F[:, :, -1], tf.expand_dims(tf.cast(mask1 == index, tf.float64), -1) * tf.gather(Edges, FFlatIndices)), -1)
            F = tf.experimental.numpy.append(F, result, axis = -1)

            # For the irregular case the recursion must return the same value until it hits a real edge
            # To achieve that the following parameters should be passed : BoxPlusOp(L1 = -value, L2 = -inf)
            isRealEdge = mask2 == (max(self.cDegrees)-(1+index))
            negationFilter = tf.expand_dims(tf.cast(isRealEdge, tf.float64), -1) * 2 - 1
            mInfFilter = (tf.expand_dims(tf.cast(~isRealEdge, tf.float64), -1) * -(1.7E+308 - 1)) + 1
            result = tf.expand_dims(self.BoxPlusOp(negationFilter * B[:, :, -1], mInfFilter * tf.gather(Edges, BFlatIndices)), -1)
            B = tf.experimental.numpy.append(B, result, axis = -1)
        return F, B

    def CalculateCMarginalsBoxPlus(self, Edges):
        F, B = self.BoxPlusDecoder(Edges, max(self.cDegrees) - 2)
        rangeIndices = np.arange(self.cDegrees.shape[0])
        rangeBatches = np.arange(self.batchSize)
        newEdges = tf.zeros_like(Edges)
        for i in np.arange(1, max(self.cDegrees) - 1, 1):
            filteredRangeIndices = rangeIndices[self.cDegrees > i]
            twoDimIndicesCN = tf.transpose([filteredRangeIndices, tf.zeros_like(filteredRangeIndices) + i])
            threeDimIndicesF = tf.transpose([tf.tile(filteredRangeIndices, [rangeBatches.shape[0]]), tf.repeat(rangeBatches, filteredRangeIndices.shape[0]), tf.tile(tf.zeros_like(filteredRangeIndices) + i-1, [rangeBatches.shape[0]])])
            threeDimIndicesB = tf.transpose([tf.tile(filteredRangeIndices, [rangeBatches.shape[0]]), tf.repeat(rangeBatches, filteredRangeIndices.shape[0]), tf.tile(tf.zeros_like(filteredRangeIndices) + max(self.cDegrees) - (i+2), [rangeBatches.shape[0]])])
            filteredF = tf.gather_nd(F, threeDimIndicesF)
            filteredB = tf.gather_nd(B, threeDimIndicesB)
            filteredCN = tf.gather_nd(self.cNodes, twoDimIndicesCN)
            newEdges += self.IndexFilter(Edges, filteredCN, self.BoxPlusOp(filteredF, filteredB))
        twoDimIndicesCN = tf.transpose([rangeIndices, tf.zeros_like(rangeIndices) + self.cDegrees - 1])
        threeDimIndicesF = tf.transpose([tf.tile(rangeIndices, [rangeBatches.shape[0]]), tf.repeat(rangeBatches, rangeIndices.shape[0]), tf.tile(self.cDegrees - 2, [rangeBatches.shape[0]])])
        threeDimIndicesB = tf.transpose([tf.tile(rangeIndices, [rangeBatches.shape[0]]), tf.repeat(rangeBatches, rangeIndices.shape[0]), tf.tile(tf.zeros_like(rangeIndices) + max(self.cDegrees) - 2, [rangeBatches.shape[0]])])
        newEdges += self.IndexFilter(Edges, tf.gather(self.cNodes, 0, axis = 1), tf.gather_nd(B, threeDimIndicesB))
        newEdges += self.IndexFilter(Edges, tf.gather_nd(self.cNodes, twoDimIndicesCN), tf.gather_nd(F, threeDimIndicesF))
        
        flatIndices = tf.reshape(self.cNodes, [-1])
        newValues = tf.gather(newEdges, flatIndices) * tf.cast(tf.repeat(self.syndrome, repeats=self.cDegrees, axis = 0), dtype=tf.float64)
        return self.IndexFilter(newEdges, flatIndices, newValues)
        #newEdges = tf.clip_by_value(newEdges, clip_value_min=-30, clip_value_max=1e200)
        #return newEdges
    
    def phiOp(self, x):
        mask = tf.cast(x < 30,  tf.float64)
        exponent = tf.keras.activations.exponential(x)
        phiResult = tf.math.log( (exponent + 1.0) / (exponent - 1.0) )
        return tf.clip_by_value(phiResult, 0.0, 1.0e+200) * mask

    def CalculateCMarginalsPhi(self, Edges):
        flatIndices = tf.reshape(self.cNodes, [-1])
        sign = tf.cast(tf.gather(Edges, flatIndices) >= 0, tf.float64)*2-1
        absVal = sign * tf.gather(Edges, flatIndices)
        phiValues = self.phiOp(absVal)
        
        range = tf.experimental.numpy.arange(phiValues.shape[0])
        reshapedRange = tf.RaggedTensor.from_row_lengths(range, self.cDegrees)
        orderedPhiValues = tf.gather(phiValues, reshapedRange)
        totalSum = tf.reduce_sum(orderedPhiValues, axis=1)

        prodValues = sign * (self.phiOp(tf.repeat(totalSum, repeats=self.cDegrees ,axis=0) - phiValues))
        reshapedProdValues = tf.gather(prodValues, reshapedRange).to_tensor(default_value=1)
        
        mask = np.ones(reshapedProdValues.shape[1])
        mask[0] = 0
        for i in tf.experimental.numpy.arange(reshapedProdValues.shape[1]):
            maskedProdValues = tf.boolean_mask(reshapedProdValues, np.roll(mask, i), axis=1)
            if(i == 0):
                finalProd = tf.reduce_prod(maskedProdValues, axis=1, keepdims=True)
            else:
                finalProd = tf.concat([finalProd, tf.reduce_prod(maskedProdValues, axis=1, keepdims=True)], axis=1)

        finalProdRagged = tf.RaggedTensor.from_tensor(finalProd, lengths=self.cDegrees)
        finalProdFlat = tf.reshape(finalProdRagged, [-1, finalProdRagged.shape[-1]])
        Edges = self.IndexFilter(Edges, flatIndices, finalProdFlat)
        #Edges = tf.clip_by_value(Edges, clip_value_min=-10, clip_value_max=10)

        flatIndices = tf.reshape(self.cNodes, [-1])
        newValues = tf.gather(Edges, flatIndices) * tf.cast(tf.repeat(self.syndrome, repeats=self.cDegrees, axis = 0), dtype=tf.float64)
        return self.IndexFilter(Edges, flatIndices, newValues)

    def CalculateCMarginals(self, Edges, Syndrome):
        tanhValues = tf.tanh(Edges * 0.5)
        reshapedtanhValues = tf.gather(tanhValues, self.cNodes).to_tensor(default_value=1)

        mask = np.ones(reshapedtanhValues.shape[1])
        mask[0] = 0
        for i in tf.experimental.numpy.arange(reshapedtanhValues.shape[1]):
            maskedProdValues = tf.boolean_mask(reshapedtanhValues, np.roll(mask, i), axis=1)
            if(i.numpy() == 0):
                finalProd = tf.reduce_prod(maskedProdValues, axis=1, keepdims=True)
            else:
                finalProd = tf.concat([finalProd, tf.reduce_prod(maskedProdValues, axis=1, keepdims=True)], axis=1)

        finalProdRagged = tf.RaggedTensor.from_tensor(finalProd, lengths=self.cDegrees)
        finalProdFlat = tf.reshape(finalProdRagged, [-1, finalProdRagged.shape[-1]])
        atanhValues = 2*tf.atanh(finalProdFlat)
        flatIndices = tf.reshape(self.cNodes, [-1])
        Edges = self.IndexFilter(Edges, flatIndices, atanhValues)

        newValues = tf.gather(Edges, flatIndices) * tf.cast(tf.repeat(Syndrome, repeats=self.cDegrees, axis = 0), dtype=tf.float64)
        Edges = self.IndexFilter(Edges, flatIndices, newValues)
        #Edges = tf.clip_by_value(Edges, clip_value_min=-10, clip_value_max=10)
        return Edges

    def CalculateSyndrome(self, Edges):
        return tf.reduce_sum(tf.cast(tf.gather(Edges, self.cNodes) < 0, tf.int64), axis = 1) % 2

    def CalculateSyndromeLLRs(self, LLRs):
        Edges = tf.zeros([self.EdgesCount, LLRs.shape[1]], dtype=tf.float64)
        Edges = self.FillEdges(Edges, LLRs)
        return self.CalculateSyndrome(Edges)

    def ContinueCalculation(self, decodedLLRs, Edges, iteration, loss):
        if(iteration == 0):
            return True
        elif(iteration >= self.lMax):
            return False
        else:
            return np.sum(1 * np.logical_not(self.terminated))

    def CalculateTermination(self, decodedLLRs, Edges = 0):
        if(self.isSyndromeBased):
            decodedSynd = self.CalculateSyndromeLLRs(decodedLLRs)
            originalSynd = 1 * np.array(self.syndrome < 0)
            test = np.sum(np.abs(originalSynd - decodedSynd), axis=0)
            self.terminated = np.logical_or(self.terminated, np.sum(np.abs(originalSynd - decodedSynd), axis=0) == 0)
        else:
            decodedSynd = self.CalculateSyndrome(Edges)
            self.terminated = np.logical_or(self.terminated, np.sum(decodedSynd, axis=0) == 0)

    def BeliefPropagationIt(self, decodedLLRs, Edges, iteration, loss):
        filteredEdges = tf.boolean_mask(Edges, tf.logical_not(self.terminated), axis=1)
        filteredLLRs = tf.boolean_mask(self.LLRs, tf.logical_not(self.terminated), axis=1)
        filteredSyndrome = tf.boolean_mask(self.syndrome, tf.logical_not(self.terminated), axis=1)

        filteredEdges = self.CalculateCMarginals(filteredEdges, filteredSyndrome) # Check nodes
        filteredEdgesTest = filteredEdges.numpy()
        filteredEdges = tf.clip_by_value(filteredEdges, clip_value_min=-100, clip_value_max=100)
        filteredEdgesTest2 = filteredEdges.numpy()
        filteredEdges, filteredDecodedLLRs = self.CalculateVMarginals(filteredEdges, iteration, filteredLLRs) # Variable nodes
        iteration += 1
        
        indices = tf.expand_dims(tf.boolean_mask(tf.range(Edges.shape[1]), tf.logical_not(self.terminated)), axis=-1)
        Edges = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(Edges), indices, tf.transpose(filteredEdges)))
        decodedLLRs = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(decodedLLRs), indices, tf.transpose(filteredDecodedLLRs)))
        
        condition = tf.cast(tf.logical_not(tf.cast(self.decimationMask, bool)), tf.float64)
        Edges = Edges * condition + self.decimationMask
        Edges = tf.clip_by_value(Edges, clip_value_min=-100, clip_value_max=100)
        condition2 = tf.cast(tf.logical_not(tf.cast(self.VNMaskValues, bool)), tf.float64)
        decodedLLRs = decodedLLRs * condition2 + self.VNMaskValues

        conditionTest = condition.numpy()
        condition2Test = condition2.numpy()

        terminatedBefore = self.terminated
        self.CalculateTermination(decodedLLRs, Edges)
        newTerminated = np.logical_xor(terminatedBefore, self.terminated)
        averagingScale = newTerminated * (1/iteration) + np.logical_not(newTerminated)

        loss += tf.where(tf.logical_not(terminatedBefore), self.LossFunction(decodedLLRs), 0)
        loss *= averagingScale
        
        

        # indices = tf.expand_dims(tf.boolean_mask(tf.range(decodedLLRs.shape[1]), (newTerminated)), axis=-1)
        # newCodeWord = tf.cast(tf.boolean_mask(decodedLLRs, (newTerminated), axis=1) < 0, tf.int32)
        # self.decodedWords = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(self.decodedWords), indices, tf.transpose(newCodeWord)))
        
        # terminatedBefore2 = self.terminated
        # newDecodedWords = tf.cast(self.ErasurePropagationOp(decodedLLRs), tf.int32)
        # newTerminated2 = np.logical_xor(terminatedBefore2, self.terminated)
        
        # # terminatedBefore2 = self.terminated
        # # newDecodedWords = self.ParticleSpread(decodedLLRs, 100)
        # # newTerminated2 = np.logical_xor(terminatedBefore2, self.terminated)

        # indices = tf.expand_dims(tf.boolean_mask(tf.range(decodedLLRs.shape[1]), (newTerminated2)), axis=-1)
        # newCodeWord2 = tf.boolean_mask(newDecodedWords, (newTerminated2), axis=1)
        # self.decodedWords = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(self.decodedWords), indices, tf.transpose(newCodeWord2)))
        
        return decodedLLRs, Edges, iteration, loss

    def ErasurePropagationIt(self, decodedWord, Edges, iteration, loss):
        filteredEdges = tf.boolean_mask(Edges, tf.logical_not(self.terminated), axis=1)
        filteredDecodedWord = tf.boolean_mask(decodedWord, tf.logical_not(self.terminated), axis=1)
        originalSynd = tf.cast(self.syndrome < 0, tf.float64)
        filteredSyndrome = tf.boolean_mask(originalSynd, tf.logical_not(self.terminated), axis=1)

        filteredEdges = self.CalculateCErasures(filteredEdges, filteredSyndrome) # Check nodes
        filteredEdges, filteredDecodedWords = self.CalculateVErasures(filteredEdges, filteredDecodedWord) # Variable nodes
        iteration += 1
        
        indices = tf.expand_dims(tf.boolean_mask(tf.range(Edges.shape[1]), tf.logical_not(self.terminated)), axis=-1)
        Edges = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(Edges), indices, tf.transpose(filteredEdges)))
        decodedWord = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(decodedWord), indices, tf.transpose(filteredDecodedWords)))
        
        isErasure, isNotErasure = IsFloat(tf.reduce_sum(decodedWord, axis=0))
        decodedSynd = (self.H @ tf.cast(decodedWord, tf.int32))%2
        originalSynd = 1 * np.array(self.syndrome < 0)
        test = isNotErasure * (np.sum(np.abs(originalSynd - decodedSynd), axis=0) == 0)
        self.terminated = np.logical_or(self.terminated, isNotErasure * (np.sum(np.abs(originalSynd - decodedSynd), axis=0) == 0))

        return decodedWord, Edges, iteration, 0
    
    def DecimateNodes(self, decodedLLRs, decimationCounter, localCounter):
        VNMaskValuesTemp = self.VNMaskValues
        while(True):
            VNMask = tf.logical_not(tf.cast(VNMaskValuesTemp, bool))
            test = (tf.cast(VNMaskValuesTemp, bool)).numpy()
            filteredDecodedLLRs = tf.where(VNMask, decodedLLRs, 1e8)
            absFilteredDecodedLLRs = tf.abs(filteredDecodedLLRs) #tf.abs(tf.ragged.boolean_mask(decodedLLRs, VNMask).to_tensor(default_value=0))
            vNodeIndex = tf.argmin(absFilteredDecodedLLRs, axis=0, output_type=tf.int32)
            indices = tf.stack([vNodeIndex, tf.range(vNodeIndex.shape[0])], axis=1)
            updates1 = tf.gather_nd(filteredDecodedLLRs, indices)
            decimationCounter = decimationCounter #+ tf.cast(abs(updates1) >= 10, tf.float64)
            
            updatesFactor = 1
            if(localCounter != 0):
                modOp = localCounter % 2
                updatesFactor = pow(-1, modOp) 
            updates = updates1 * 1000 * updatesFactor
            VNMaskValuesTemp = tf.tensor_scatter_nd_update(VNMaskValuesTemp, indices, updates)
            #if(self.RankCheck(vNodeIndex)):
            if(localCounter != 0):
                localCounter = (localCounter - modOp)/2
            break

        self.VNMaskValues = tf.tensor_scatter_nd_update(self.VNMaskValues, indices, updates)

        flatIndices = tf.reshape(self.vNodes, [-1])
        newValues =  tf.repeat(self.VNMaskValues, repeats=self.vDegrees, axis = 0)
        self.decimationMask = self.IndexFilter(self.decimationMask, flatIndices, newValues)
        return decimationCounter, localCounter
    
    def DecimateNodesOld(self, decodedLLRs, CNMask):
        Edges = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
        Edges = self.FillEdges(Edges, decodedLLRs)

        qBitValue = self.Channel.CalculateSingleLLRTF(0)
        for i in range(self.batchSize):
            filteredNodes = tf.ragged.boolean_mask(self.cNodes, CNMask[:, i])
            checkNodeIndex = tf.argmax(tf.reduce_mean(tf.gather(Edges[:, i], filteredNodes), axis = 1), axis=0)
            CNMask = tf.tensor_scatter_nd_update(CNMask, [[checkNodeIndex, i]], [False])

            treeIndex = tf.constant(i, dtype=tf.int64)
            connectedVNodes = self.cNodes[checkNodeIndex, :]
            VNodeIndex = tf.where(tf.equal(connectedVNodes[0], self.vNodes))[0][0]
            connectedCNodes = self.vNodes[VNodeIndex, :]

            repeatedTreeIndex = tf.repeat(treeIndex, repeats=connectedCNodes.shape[0])
            repeatedCNodeValue = tf.repeat(qBitValue, repeats=connectedCNodes.shape[0])
            finalIndices = tf.stack([connectedCNodes, repeatedTreeIndex], axis=1)
            self.decimationMask = tf.tensor_scatter_nd_update(self.decimationMask, finalIndices, repeatedCNodeValue)
        return CNMask

    def TrainNBP(self):
        self.GenerateCodeWords()
        EdgesIni = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
        EdgesIni = self.FillEdges(EdgesIni, self.LLRs)
        self.decimationMask = tf.zeros_like(EdgesIni)
        self.VNMaskValues = tf.zeros([self.vNodes.shape[0], self.batchSize], dtype=tf.float64)
        
        self.terminated = np.zeros(EdgesIni.shape[1], dtype=bool)
        self.decodedWords = tf.zeros_like(self.LLRs, dtype=tf.int32)
        # parameters: [decodedLLRs, iteration, loss]
        decodedLLRs, Edges, iteration, loss = \
                            tf.while_loop(\
                            self.ContinueCalculation,\
                            self.BeliefPropagationIt,\
                            [tf.ones([self.vNodes.shape[0], self.batchSize], dtype=tf.float64) * -1,\
                            EdgesIni,\
                            tf.constant(0,dtype=tf.int32),\
                            tf.constant(0.0, dtype=tf.float64)])
        #totalLoss = loss #loss/tf.cast(iteration, tf.float64)
        tf.print(loss)
        return loss

    def NeuralBeliefPropagationOp(self):
        channelOutput, channelInput = self.GenerateCodeWords()
        EdgesIni = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
        EdgesIni = self.FillEdges(EdgesIni, self.LLRs)
        self.decimationMask = tf.zeros_like(EdgesIni)
        self.VNMaskValues = tf.zeros([self.vNodes.shape[0], self.batchSize], dtype=tf.float64)
        
        self.channelOutput = channelOutput

        self.terminated = np.zeros(EdgesIni.shape[1], dtype=bool)
        self.decodedWords = tf.zeros_like(self.LLRs, dtype=tf.int32)
        # parameters: [decodedLLRs, iteration, loss]
        decodedLLRs, Edges, iteration, loss = \
            tf.nest.map_structure(tf.stop_gradient,\
                            tf.while_loop(\
                            self.ContinueCalculation,\
                            self.BeliefPropagationIt,\
                            [tf.ones([self.vNodes.shape[0], self.batchSize], dtype=tf.float64) * -1,\
                            EdgesIni,\
                            tf.constant(0,dtype=tf.int32),\
                            tf.constant(0.0, dtype=tf.float64)]))
        
        #tf.cast(decodedLLRs < 0, tf.int64)

        return Edges, decodedLLRs, channelOutput, channelInput

    def ErasurePropagationOp(self, decodedLLRs):
        # originalLMax = self.lMax
        # self.lMax = 1

        EdgesIni, decodedWords = self.ErasureTransform(decodedLLRs)
        test1 = decodedWords.numpy()
        error = np.abs(test1 - self.channelOutput)
        decodedLLRsTest = decodedLLRs.numpy()

        # parameters: [decodedWords, iteration, loss]
        decodedWords, Edges, iteration, loss = \
            tf.nest.map_structure(tf.stop_gradient,\
                            tf.while_loop(\
                            self.ContinueCalculation,\
                            self.ErasurePropagationIt,\
                            [decodedWords,\
                            EdgesIni,\
                            tf.constant(0,dtype=tf.int32),\
                            tf.constant(0.0, dtype=tf.float64)]))

        #self.lMax = originalLMax
        return decodedWords

    def ParticleBeliefPropagation(self):
        channelOutput, channelInput = self.GenerateCodeWords()
        EdgesIni = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
        EdgesIni = self.FillEdges(EdgesIni, self.LLRs)
        self.decimationMask = tf.zeros_like(EdgesIni)
        self.VNMaskValues = tf.zeros([self.vNodes.shape[0], self.batchSize], dtype=tf.float64)
        
        self.terminated = np.zeros(EdgesIni.shape[1], dtype=bool)
        self.decodedWords = tf.zeros_like(self.LLRs, dtype=tf.int32)
        # parameters: [decodedLLRs, iteration, loss]
        decodedLLRs, Edges, iteration, loss = \
            tf.nest.map_structure(tf.stop_gradient,\
                            tf.while_loop(\
                            self.ContinueCalculation,\
                            self.BeliefPropagationIt,\
                            [tf.ones([self.vNodes.shape[0], self.batchSize], dtype=tf.float64) * -1,\
                            EdgesIni,\
                            tf.constant(0,dtype=tf.int32),\
                            tf.constant(0.0, dtype=tf.float64)]))
    
        # decodedLLRsTest = decodedLLRs.numpy()
        # np.savetxt("decodedLLRs", decodedLLRsTest)
        # originalSynd = 1 * np.array(self.syndrome < 0)
        # np.savetxt("originalSynd", originalSynd)

        #self.ParticleSpread(decodedLLRs, 100)

        return Edges, self.decodedWords, channelOutput, channelInput

    def DecimatedBeliefPropagation(self, ResultsCalculator):
        channelOutput, channelInput = self.GenerateCodeWords()
        NumberOfDecimatedNodesPerBP = 1
        NumberOfBPOperations = 6 #192 #45
        ResultsCalculator.SetUpdate(False)
        currentDecimationCounter = 0
        localCounter = 0
        decimationCounter = 0
        self.terminated = np.zeros(self.batchSize, dtype=bool)
        self.decodedWords = tf.zeros_like(self.LLRs, dtype=tf.int32)

        while(True):
            self.decimationMask = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
            self.VNMaskValues = tf.zeros([self.vNodes.shape[0], self.batchSize], dtype=tf.float64)
            self.FRMat = tf.zeros([self.Hx.shape[0], 0], tf.int32)
            
            for i in range(NumberOfBPOperations + 1) :
                EdgesIni = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
                EdgesIni = self.FillEdges(EdgesIni, self.LLRs)
                condition = tf.cast(tf.logical_not(tf.cast(self.decimationMask, bool)), tf.float64)
                EdgesIni = EdgesIni * condition + self.decimationMask
                EdgesIni = tf.clip_by_value(EdgesIni, clip_value_min=-100, clip_value_max=100)
                test = EdgesIni.numpy()

                # parameters: [decodedLLRs, iteration, loss]
                decodedLLRs, Edges, iteration, loss = \
                    tf.nest.map_structure(tf.stop_gradient,\
                                    tf.while_loop(\
                                    self.ContinueCalculation,\
                                    self.BeliefPropagationIt,\
                                    [tf.ones([self.vNodes.shape[0], self.batchSize], dtype=tf.float64) * -1,\
                                    EdgesIni,\
                                    tf.constant(0,dtype=tf.int32),\
                                    tf.constant(0.0, dtype=tf.float64)]))
                
                if(decimationCounter > currentDecimationCounter):
                    currentDecimationCounter = decimationCounter
                    localCounter = decimationCounter
                    break
                if(np.sum(1 * np.logical_not(self.terminated)) > 0):
                    if(i != NumberOfBPOperations):
                        errorWord = ((decodedLLRs < 0).numpy() + channelOutput + channelInput) % 2 
                        ResultsCalculator.CaclulateErrorRate(errorWord, self.terminated)
                        for _ in range(NumberOfDecimatedNodesPerBP):
                            decimationCounter, localCounter = self.DecimateNodes(decodedLLRs, currentDecimationCounter, localCounter)
                            if(decimationCounter > currentDecimationCounter):
                                print("Stopped Decimating at :" + str(decimationCounter))
                                break
                        print("Decimated Nodes : " + str((i + 1) * NumberOfDecimatedNodesPerBP))
                    else:
                        currentDecimationCounter += 1
                        localCounter = currentDecimationCounter
                        print("Decimation Round Done")
                else:
                    break
            
            break
            #Terminate the while(True)
            if((np.sum(1 * np.logical_not(self.terminated)) == 0) or (currentDecimationCounter == (2 ** NumberOfBPOperations))):
                break

        ResultsCalculator.SetUpdate(True)
        return Edges, tf.cast(decodedLLRs < 0, tf.int64), channelOutput, channelInput

    def ErasureTransform(self, decodedLLRs):
        EdgesIni = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
        EdgesIni = self.FillEdges(EdgesIni, decodedLLRs)
        checkNodes = tf.gather(EdgesIni, self.cNodes)

        expandedCheckNode = checkNodes.to_tensor(default_value=50)
        nodesIndex = tf.argmin(tf.abs(expandedCheckNode), axis=1)
        oneHot = tf.one_hot(indices=nodesIndex , depth=expandedCheckNode.shape[1], on_value=False, off_value=True, axis=1)
        
        expandedCheckNode = tf.cast(expandedCheckNode < 0, tf.float64)
        erasured = tf.where(oneHot, expandedCheckNode, ERASURE)
        raggedCheckNodes = tf.RaggedTensor.from_tensor(erasured, lengths=self.cDegrees)

        flatCheckNodes = tf.reshape(raggedCheckNodes, [-1, raggedCheckNodes.shape[-1]])
        flatIndices = tf.reshape(self.cNodes, [-1])
        Edges = self.IndexFilter(EdgesIni, flatIndices, flatCheckNodes)

        _, decodedWords = self.CalculateVErasures(Edges, tf.ones([self.vNodes.shape[0], self.batchSize], dtype=tf.float64) * ERASURE)

        return Edges, decodedWords

    def ParticleSpread(self, decodedLLRs, particlesCount):
        decodedWords = tf.cast(decodedLLRs < 0, tf.int32)
        oneProb = Fermi(decodedLLRs)
        for i in range(particlesCount):
            
            filteredOneProb = tf.boolean_mask(oneProb, tf.logical_not(self.terminated), axis=1)
            particles = tf.random.uniform(filteredOneProb.shape, dtype=tf.dtypes.float64)
            newCodeWord = tf.cast(particles < filteredOneProb, tf.int32)

            indices = tf.expand_dims(tf.boolean_mask(tf.range(decodedLLRs.shape[1]), tf.logical_not(self.terminated)), axis=-1)
            decodedWords = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(decodedWords), indices, tf.transpose(newCodeWord)))

            decodedSynd = (self.H @ newCodeWord)%2
            originalSynd = tf.cast(tf.boolean_mask(self.syndrome, tf.logical_not(self.terminated), axis=1) < 0, tf.int32)
            diff = tf.cast(tf.reduce_sum(tf.abs(originalSynd - decodedSynd), axis=0) == 0, tf.int32)
            terminatedBefore = self.terminated
            self.terminated = tf.cast(tf.transpose(tf.tensor_scatter_nd_max(tf.cast(tf.transpose(self.terminated), tf.int32), indices, tf.transpose(diff))), tf.bool)
            
            # newTerminated = np.logical_xor(terminatedBefore, self.terminated)
            # if(tf.reduce_sum(tf.cast(newTerminated, tf.int32)) > 0):
            #     waleed = 0

            if(tf.reduce_sum(tf.cast(tf.logical_not(self.terminated), tf.int32)) == 0):
                break

        return decodedWords

    def RankCheck(self, nodeIndex):
        test2 = nodeIndex.numpy()[0]
        test = tf.gather(self.Hx, [test2], axis=1)
        newFRMat = tf.concat([self.FRMat, test], axis=1)

        if(rank(self.FRMat.numpy()) < rank(newFRMat.numpy())):
            self.FRMat = newFRMat
            return True
        return False

    def GenerateCodeWords(self):
        LLRs, channelOutput, channelInput = self.Channel.GenerateLLRs(True)
        
        if(self.isSyndromeBased):
            self.LossFunction.SetLabels((channelOutput + channelInput)%2)
            self.SetSyndrome(LLRs)
            LLRs, _ , _ = self.Channel.GenerateLLRs(False)
        else:
            self.LossFunction.SetLabels(channelInput)

        self.SetLLRs(LLRs)
        return channelOutput, channelInput

    # def DecimateNodes(self, decodedLLRs, CNMask):
    #     Edges = tf.zeros([self.EdgesCount, self.batchSize], dtype=tf.float64)
    #     Edges = self.FillEdges(Edges, decodedLLRs)

    #     qBitValue = self.Channel.CalculateSingleLLRTF(0)
    #     for i in range(self.batchSize):
    #         filteredNodes = tf.ragged.boolean_mask(self.cNodes, CNMask[:, i])
    #         checkNodeIndex = tf.argmax(tf.reduce_mean(tf.gather(Edges[:, i], filteredNodes), axis = 1), axis=0)
    #         CNMask = tf.tensor_scatter_nd_update(CNMask, [[checkNodeIndex, i]], [False])

    #         treeIndex = tf.constant(i, dtype=tf.int64)
    #         connectedVNodes = self.cNodes[checkNodeIndex, :]
    #         VNodeIndex = tf.where(tf.equal(connectedVNodes[0], self.vNodes))[0][0]
    #         connectedCNodes = self.vNodes[VNodeIndex, :]

    #         repeatedTreeIndex = tf.repeat(treeIndex, repeats=connectedCNodes.shape[0])
    #         repeatedCNodeValue = tf.repeat(qBitValue, repeats=connectedCNodes.shape[0])
    #         finalIndices = tf.stack([connectedCNodes, repeatedTreeIndex], axis=1)
    #         self.decimationMask = tf.tensor_scatter_nd_update(self.decimationMask, finalIndices, repeatedCNodeValue)
    #     return CNMask