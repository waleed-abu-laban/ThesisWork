from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

lMax = 60
Ns = [256, 1024, 2048, 4096, 8192]
ALLBERpy = [[]]
ALLBERc = [[]]
ALLEbN0s = [[]]
for N in Ns:
    NK = N//2
    BERpy = []
    BERc = []
    EbN0s = []
    fpy = open("results7l\LDPC_(3,6)_n"+ str(N) + "_k" + str(NK) + "_BP_Random_" + str(lMax) + "it_AWGN_Fastpy.txt", "r")
    fc = open("results7l\LDPC_(3,6)_n"+ str(N) + "_k" + str(NK) + "_BP_Random_" + str(lMax) + "it_AWGNc.txt", "r")
    fr = open("results7l\LDPC_(3,6)_n"+ str(N) + "_k" + str(NK) + "_BP_Random_" + str(lMax) + "it_AWGN_EbNo.txt", "r")
    for i in fpy:
        BERpy.append(float(i))
    for i in fc:
        BERc.append(float(i))
    for i in fr:
        EbN0s.append(float(i))
    ALLBERpy.append(BERpy)
    ALLBERc.append(BERc)
    ALLEbN0s.append(EbN0s)
ALLBERpy = ALLBERpy[1:len(ALLBERpy)]
ALLBERc = ALLBERc[1:len(ALLBERc)]
ALLEbN0s = ALLEbN0s[1:len(ALLEbN0s)]

for i in range(5):
    plt.figure()
    plt.semilogy(ALLEbN0s[i], ALLBERpy[i], color = 'r', marker = 'o')
    plt.semilogy(ALLEbN0s[i], ALLBERc[i], color = 'b', marker = 's')
    plt.grid(True)
    plt.title("Regular LDPC (3, 6), 10 iterations, n = " + str(Ns[i]) + ", R = 0.5")
    #plt.title("Regular LDPC (3, 6), 10 iterations")
    plt.ylabel("Bit error rate")
    plt.xlabel("E_b/N_o")

    resultsPy = mpatches.Patch(color = 'r', label = 'python results')
    resultsC = mpatches.Patch(color = 'b', label = 'C results')
    plt.legend(handles=[resultsPy,resultsC])

plt.show()