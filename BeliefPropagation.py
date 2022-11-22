import numpy as np

class BPInstance:
    def __init__(self, CodeDescriptor):
        self.Edges = CodeDescriptor.Edges
        self.vNodes = CodeDescriptor.vNodes
        self.cNodes = CodeDescriptor.cNodes
        self.vDegrees = CodeDescriptor.vDegrees
        self.cDegrees = CodeDescriptor.cDegrees

    def ClearEdges(self):
        self.Edges *= 0
    
    def SetLLRs(self, LLRs):
        self.LLRs = LLRs

    def SetSyndrome(self, syndrome = -1):
        if np.isscalar(syndrome):
            self.syndrome = np.ones(self.cNodes.shape[0])
        else:
            self.syndrome = syndrome
    
    def ThresholdClip(self):
        threshold = 30
        self.Edges[self.Edges < -threshold] = -threshold
        self.Edges[self.Edges > threshold] = threshold

    def CalculateVMarginals(self):
        sum = self.LLRs + np.sum(self.Edges[np.array(self.vNodes)], axis = 1) # Calculate Variable Belief
        self.Edges[np.array(self.vNodes).flatten()] = np.repeat(sum, self.vNodes.shape[1]) - self.Edges[np.array(self.vNodes).flatten()]
        self.Edges[-1] = 0
        #ThresholdClip()
        return sum

    def BoxPlusOp(self, L1, L2):
        maxThreshold = 1e30
        term1 =(1 + np.exp(L1 + L2))
        term1[term1 < -maxThreshold] = -maxThreshold
        term1[term1 > maxThreshold] = maxThreshold
        term2 = (np.exp(L1) + np.exp(L2))
        term2[term2 < -maxThreshold] = -maxThreshold
        term2[term2 > maxThreshold] = maxThreshold
        return np.log(term1 / term2)

    def BoxPlusDecoder(self, index):
        if(index == 0):
            F = np.expand_dims(self.Edges[np.array(self.cNodes[:, index])], axis = 1)
            B = np.expand_dims(self.Edges[np.array(self.cNodes[range(self.cNodes.shape[0]), self.cDegrees -(1+index)])], axis = 1)
        else:
            F, B = self.BoxPlusDecoder(index - 1)

            result = np.expand_dims(self.BoxPlusOp(F[:, -1], self.Edges[np.array(self.cNodes[:, index])]), axis = 1)
            F = np.append(F, result, axis = 1)
            result = np.expand_dims(self.BoxPlusOp(B[:, -1], self.Edges[np.array(self.cNodes[range(self.cNodes.shape[0]), self.cDegrees -(1+index)])]), axis = 1)
            B = np.append(B, result, axis = 1)
        return F, B

    def CalculateCMarginals(self):
        F, B = self.BoxPlusDecoder(self.cNodes.shape[1] - 2)
        for i in np.arange(1, self.cNodes.shape[1] - 1, 1):
            self.Edges[np.array(self.cNodes[:, i])] = self.BoxPlusOp(F[:, i-1], B[range(B.shape[0]), self.cDegrees - (i+2)])
        self.Edges[np.array(self.cNodes[:, 0])] = B[range(B.shape[0]), self.cDegrees - 2]
        self.Edges[np.array(self.cNodes[range(self.cNodes.shape[0]), self.cDegrees - 1])] = F[range(F.shape[0]), self.cDegrees - 2]
        self.Edges[np.array(self.cNodes).flatten()] *= np.repeat(self.syndrome, self.cNodes.shape[1])
        #ThresholdClip()

    def CalculateSyndrome(self):
        # should be modified in case of quantum, x and z parts should be swaped?
        return np.sum(1 * (self.Edges[np.array(self.cNodes)] < 0), axis = 1) % 2

    def IsCodeWord(self):
        return np.sum(self.CalculateSyndrome()) == 0 # check if CodeWord

    def BeliefPropagation(self, lMax):
        decodedLLRs = self.CalculateVMarginals() # Initialize
        for _ in range(lMax):
            self.CalculateCMarginals() # Check nodes
            decodedLLRs = self.CalculateVMarginals() # Variable nodes
            if(self.IsCodeWord()):
                break
        decodedWord = 1*(decodedLLRs < 0)
        return decodedLLRs, decodedWord
