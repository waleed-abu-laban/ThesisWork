import numpy as np

class BaseCalculator:
    def __init__(self, batchSizeTest, maxFrameErrorCount, maxFrames):
        self.Reset()
        self.batchSizeTest = batchSizeTest
        self.maxFrameErrorCount = maxFrameErrorCount
        self.maxFrames = maxFrames
        self.updateOn = True

    def Reset(self):
        self.errorCountTotal = 0
        self.frameCount = 0
        self.frameErrorCount = 0

    def Continue(self):
        return (self.frameErrorCount < self.maxFrameErrorCount) and (self.frameCount < self.maxFrames)
    
    def SetParameter(self, parameter):
        self.parameter = parameter

    def GetParameter(self):
        return self.parameter

    def SetUpdate(self, updateOn):
        self.updateOn = updateOn

class QuantumCalculator(BaseCalculator):
    def __init__(self, batchSizeTest, maxFrameErrorCount, maxFrames, Hortho):
        super().__init__(batchSizeTest, maxFrameErrorCount, maxFrames)
        self.Hortho = Hortho

    def CaclulateErrorRate(self, errorWord, termination):
        frameCountCurrent = self.frameCount + self.batchSizeTest
        if(self.updateOn):
            self.frameCount = frameCountCurrent

        # Flagged
        filterDataSynd = np.logical_not(termination)

        # Unflagged
        #errorWord = (decodedWord + channelOutput) % 2
        errorWordFiltered = errorWord[:, termination]
        logicalOpErrors = np.sum(self.Hortho @ errorWordFiltered % 2, axis=0)
        filterDataHmat = logicalOpErrors > 0
        
        #errorCountTotal += np.sum(np.dot(self.Hortho, errorWord) % 2)
        frameErrorCountCurrent = self.frameErrorCount + (np.sum(1*(filterDataSynd))) #+ np.sum(1*(filterDataHmat))) #np.sum(1*(test > 0)) #
        if(self.updateOn):
            self.frameErrorCount = frameErrorCountCurrent
        errorRate = frameErrorCountCurrent / frameCountCurrent

        # print the data to keep track -------------------------------------------------
        print(self.parameter, frameCountCurrent, errorRate, frameErrorCountCurrent)

        return errorRate

class LogicalQuantumCalculator(BaseCalculator):
    def __init__(self, batchSizeTest, maxFrameErrorCount, maxFrames, lx, lz):
        super().__init__(batchSizeTest, maxFrameErrorCount, maxFrames)
        self.lx = lx
        self.lz = lz

    def CaclulateErrorRate(self, errorWord, termination):
        frameCountCurrent = self.frameCount + self.batchSizeTest
        if(self.updateOn):
            self.frameCount = frameCountCurrent

        # Flagged
        filterDataSynd = np.logical_not(termination)

        # Unflagged
        #errorWord = (decodedWord + channelOutput) % 2
        errorWordFiltered = errorWord[:, termination]
        half = errorWordFiltered.shape[0] // 2
        residual_x = errorWordFiltered[half:, :]
        residual_z = errorWordFiltered[:half, :]

        failx = np.sum(self.lz@residual_x % 2, axis=0) > 0
        failz = np.sum(self.lx@residual_z % 2, axis=0) > 0
        frameErrorCountCurrent = self.frameErrorCount + np.sum(1*(np.logical_or(failx, failz))) + np.sum(1*(filterDataSynd))
        if(self.updateOn):
            self.frameErrorCount = frameErrorCountCurrent
        errorRate = frameErrorCountCurrent/frameCountCurrent
        #errorRate = 1.0 - (1-errorRate)**(1/28)

        # print the data to keep track -------------------------------------------------
        print(self.parameter, frameCountCurrent, errorRate, frameErrorCountCurrent)

        return errorRate

class ClassicalCalculator(BaseCalculator):
    def __init__(self, batchSizeTest, maxFrameErrorCount, maxFrames, N = 0):
        super().__init__(batchSizeTest, maxFrameErrorCount, maxFrames)
        self.N = N

    def CaclulateErrorRate(self, errorWord):
        frameCountCurrent = self.frameCount + self.batchSizeTest
        if(self.updateOn):
            self.frameCount = frameCountCurrent

        errorCount = np.sum(errorWord, axis=0)
        frameErrorCountCurrent = self.frameErrorCount + np.sum(1*(errorCount > 0))
        if(self.updateOn):
            self.frameErrorCount = frameErrorCountCurrent

        if(self.N > 0):
            errorCountTotalCurrent = self.errorCountTotal + np.sum(errorCount)
            if(self.updateOn):
                self.errorCountTotal = errorCountTotalCurrent
            errorRate = errorCountTotalCurrent / self.N / frameCountCurrent #BER
        else:
            errorRate = frameErrorCountCurrent / frameCountCurrent #BLER
        
        # print the data to keep track -------------------------------------------------
        print(self.parameter, frameCountCurrent, errorRate, frameErrorCountCurrent)

        return errorRate
        


# # calculate the error rates ---------------------------------------------------
# # ******************* Quantum *******************
# if(IsQuantum):
#     decodedWord = decodedWord.numpy()
#     frameCount += batchSizeTest
#     if(True):
#         # Flagged
#         filterDataSynd = np.logical_not(NBPObject.terminated)

#         # Unflagged
#         errorWord = (decodedWord + channelOutput) % 2
#         errorWordFiltered = errorWord[:, NBPObject.terminated]
#         logicalOpErrors = np.sum(Hortho @ errorWordFiltered % 2, axis=0)
#         filterDataHmat = logicalOpErrors > 0
        
#         errorCountTotal += np.sum(np.dot(Hortho, errorWord) % 2)
#         frameErrorCount += (np.sum(1*(filterDataSynd)) + np.sum(1*(filterDataHmat))) #np.sum(1*(test > 0)) #
#         errorRate = frameErrorCount / frameCount
#     else:
#         lx = np.loadtxt(generatorMatrixPaths[0])
#         lz = np.loadtxt(generatorMatrixPaths[1])

#         half = decodedWord.shape[0] // 2
#         ex = decodedWord[half:, :]
#         ez = decodedWord[:half, :]

#         residual_x = (ex + channelOutput[half:, :]) % 2
#         residual_z = (ez + channelOutput[:half, :]) % 2

#         failx = np.sum(lz@residual_x % 2, axis=0) > 0
#         failz = np.sum(lx@residual_z % 2, axis=0) > 0
#         bp_success_count += np.sum(1*np.logical_not(np.logical_or(failx, failz)))

#         errorRate = 1-bp_success_count/frameCount

#         #errorRate = 1.0 - (1-bp_logical_error_rate)**(1/28)
# # ******************* Classical *******************
# else:
#     if(isSyndromeBased):
#         errorWord = (decodedWord + channelOutput + channelInput) % 2
#     else:
#         errorWord = (decodedWord + channelInput) % 2
#     errorCount = np.sum(errorWord, axis=0)
#     errorCountTotal += np.sum(errorCount)
#     frameCount += batchSizeTest
#     frameErrorCount += np.sum(1*(errorCount > 0))
#     if(isSyndromeBased):
#         errorRate = frameErrorCount / frameCount
#     else:
#         errorRate = errorCountTotal / N / frameCount
