import numpy as np
from scipy.sparse import csr_matrix

def GenerateCode(N, K, Weights):
    N2 = N//2
    weightsPos = np.random.choice(N2, Weights, replace=False)
    A = np.zeros(N2, dtype=int)
    A[weightsPos] = 1
    C = CreateCyclicMatrix(np.roll(A,-1), 0)
    H0 = np.concatenate((C, C.transpose()), axis = 1)
    rawsToKeep = np.sort(np.random.choice(N2, N2 - K//2, replace=False))
    H0 = H0[rawsToKeep]
    zeroMat = np.zeros_like(H0)
    return np.concatenate((np.concatenate((H0, zeroMat), axis = 1), np.concatenate((zeroMat, H0), axis = 1)), axis = 0)

def CreateCyclicMatrix(A, i):
    if(i < len(A)-1):
        i+=1
        return np.append(CreateCyclicMatrix(np.roll(A,-1), i), np.expand_dims(A, axis = 0), axis = 0)
    else:
        return np.expand_dims(A, axis = 0)

def SparseHmatrix(H, fileName):
    A = csr_matrix(H)
    with open(fileName, "w") as f:
        np.savetxt(f, A, fmt='%d')
    return A

def FormatHmatrix(H, fileName):
    open(fileName, "w")
    with open(fileName, "a") as f:
        # .alist file conditions:

        # 1) The first line contains the size of the matrix as columns and rows.
        np.savetxt(f, np.array([H.shape[1]]), fmt='%d', newline=" ", delimiter="")
        np.savetxt(f, np.array([H.shape[0]]), fmt='%d')
        indices = np.where(H == 1)

        # 2) The second line contains the maximum number of ones per column followed by the maximum number of ones per row.

        # 3) Then follows a line that contains, for every column and row of the matrix, the number of ones in this column and row.

        # 4) The following lines contain, for each column and row, the positions of the ones in that column and row, A 0 indicates an unused entry.
        
        maxOnesNum = np.zeros(2, dtype=int)
        onesNumVec = np.array([np.zeros(H.shape[1], dtype=int), np.zeros(H.shape[0], dtype=int)])
        for j in range(2):
            index0 = (j+1) % 2
            index1 = (j) % 2
            maxNodeKey = indices[index0].max()
            for i in range(maxNodeKey + 1):
                ones = indices[index0] == i
                onesNum = np.sum(ones)
                onesNumVec[j][i] = onesNum
                maxOnesNum[j] = max(onesNum, maxOnesNum[j])

        np.savetxt(f, np.array([maxOnesNum[0]]), fmt='%d', newline=" ", delimiter="")
        np.savetxt(f, np.array([maxOnesNum[1]]), fmt='%d')
        np.savetxt(f, onesNumVec[0], fmt='%d', newline=" ", delimiter="")
        f.write("\n")
        np.savetxt(f, onesNumVec[1], fmt='%d', newline=" ", delimiter="")

        for j in range(2):
            index0 = (j+1) % 2
            index1 = (j) % 2
            maxNodeKey = indices[index0].max()
            for i in range(maxNodeKey + 1):
                f.write("\n")
                np.savetxt(f, indices[index1][indices[index0] == i] + 1, fmt='%d', newline=" ", delimiter="")

        # maxVarNodeKey = indices[1].max()
        # for i in range(maxVarNodeKey + 1):
        #     test = indices[0][indices[1] == i]
        #     f.write("\n")
        #     np.savetxt(f, test, fmt='%d', newline=" ")

        # maxCheckNodeKey = indices[0].max()
        # for i in range(maxCheckNodeKey + 1):
        #     test = indices[1][indices[0] == i]
        #     f.write("\n")
        #     np.savetxt(f, test, fmt='%d', newline=" ")



        # for nodeType in range(2):
        #     vSum = np.sum(H, axis=nodeType)
        #     vSum[vSum == 0] = -1
        #     counter1 = 0
        #     counter2 = 0
        #     lineValue = []
        #     for i in indices[nodeType]:
        #         lineValue.append(i)
        #         counter2 += 1
        #         while(vSum[counter1] == -1):
        #             f.write("\n")
        #             np.savetxt(f, np.array([-1]), fmt='%d', newline=" ")
        #             counter1 += 1
        #         if(vSum[counter1] == counter2):
        #             f.write("\n")
        #             np.savetxt(f, np.array(lineValue), fmt='%d', newline=" ")
        #             lineValue = []
        #             counter1 += 1
        #             counter2 = 0
        #     if(counter1 < vSum.shape[0]):
        #         while(vSum[counter1] == -1):
        #                 f.write("\n")
        #                 np.savetxt(f, np.array([-1]), fmt='%d', newline=" ")
        #                 counter1 += 1

H = GenerateCode(256, 32, 8)

#H = np.genfromtxt("codesQLDPC\Random_QLDPC_H.csv", delimiter=',')
#print(H)

FormatHmatrix(H, "Bicycle.alist")
np.savetxt("HMatrix_Bicycle.txt", H, fmt='%d')
#print(SparseHmatrix(H, "Bicycle"))