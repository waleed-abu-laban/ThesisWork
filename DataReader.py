import numpy as np
import tensorflow as tf

class Code:
	def __init__(self):
		self.num_edges = 0

def load_code(H_filename, G_filename):
	# parity-check matrix; Tanner graph parameters
	with open(H_filename) as f:
		# get n and m (n-k) from first line
		n,m = [int(s) for s in f.readline().split(' ')]
		k = n-m

		var_degrees = np.zeros(n).astype(np.int) # degree of each variable node
		chk_degrees = np.zeros(m).astype(np.int) # degree of each check node

		# initialize H
		H = np.zeros([m,n]).astype(np.int)
		max_var_degree, max_chk_degree = [int(s) for s in f.readline().split(' ')]
		f.readline() # ignore two lines
		f.readline()

		# create H, sparse version of H, and edge index matrices
		# (edge index matrices used to calculate source and destination nodes during belief propagation)
		var_edges = [[] for _ in range(0,n)]
		for i in range(0,n):
			row_string = f.readline().split(' ')
			var_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			var_degrees[i] = len(var_edges[i])
			H[var_edges[i], i] = 1

		chk_edges = [[] for _ in range(0,m)]
		for i in range(0,m):
			row_string = f.readline().split(' ')
			chk_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			chk_degrees[i] = len(chk_edges[i])

		d = [[] for _ in range(0,n)]
		edge = 0
		for i in range(0,n):
			for j in range(0,var_degrees[i]):
				d[i].append(edge)
				edge += 1

		u = [[] for _ in range(0,m)]
		edge = 0
		for i in range(0,m):
			for j in range(0,chk_degrees[i]):
				v = chk_edges[i][j]
				for e in range(0,var_degrees[v]):
					if (i == var_edges[v][e]):
						u[i].append(d[v][e])

		num_edges = H.sum()

	code = Code()
	code.H = H
	code.var_degrees = var_degrees
	code.chk_degrees = chk_degrees
	code.num_edges = num_edges
	code.u = u
	code.d = d
	code.n = n
	code.m = m
	code.k = k
	return code

class CodeDescriptor:
	def __init__(self):
            None

def ReadData(path, parityCode):
    if(parityCode == "LDPC"):
        f = open(path + ".txt", "r")
        NxNK = f.readline()
        NxNK = NxNK.split(" ")
        N = int(NxNK[0])
        NK = int(NxNK[1])
        edgeIndex = 0
        for i in range(N):
            onesPos = f.readline().split(" ")
            if(i == 0):
                vNodes = np.zeros([N, len(onesPos)], dtype=int) #[[0 for i in range(len(onesPos))] for j in range(N)]
                cNodesTemp = [[] for j in range(NK)]
                Edges = np.zeros(N * len(onesPos) + 1)
            for j in range(len(onesPos)):
                if(onesPos[j] == "\n"):
                    continue
                vNodes[i][j] = edgeIndex #int(onesPos[j])
                cNodesTemp[int(onesPos[j])].append(edgeIndex)
                edgeIndex += 1
        cNodes = np.array(cNodesTemp, dtype=int)
        vDegrees = vNodes.shape[1]
        cDegrees = cNodes.shape[1]
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

        vNodes = np.zeros((len(vNodesTemp), max(vDegrees)), dtype=int)
        for j in range(len(vNodesTemp)):
            vNodesRaw = np.array([])
            for i in range(max(vDegrees)):
                if(len(vNodesTemp[j]) > i):
                    vNodesRaw = np.append(vNodesRaw, vNodesTemp[j][i])
                else:
                    vNodesRaw = np.append(vNodesRaw, -1)
            vNodes[j] = vNodesRaw

        cNodes = np.zeros((len(cNodesTemp), max(cDegrees)), dtype=int)
        for j in range(len(cNodesTemp)):
            cNodesRaw = np.array([])
            for i in range(max(cDegrees)):
                if(len(cNodesTemp[j]) > i):
                    cNodesRaw = np.append(cNodesRaw, cNodesTemp[j][i])
                else:
                    cNodesRaw = np.append(cNodesRaw, -1)
            cNodes[j] = cNodesRaw

        Edges = np.zeros(num_edges + 1)

    CD = CodeDescriptor()
    CD.Edges = Edges
    CD.vNodes = vNodes
    CD.cNodes = cNodes
    CD.vDegrees = vDegrees
    CD.cDegrees = cDegrees
    return CD

def ReadLLRs(f, N):
    LLRs = np.array([])
    for i in range(N):
        LLRs = np.append(LLRs, float(f.readline()))
    return LLRs

def ReadDataTF(path, parityCode):
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
                Edges = tf.zeros(N * len(onesPos), dtype=tf.float64)
            for j in range(len(onesPos)):
                vNodesTemp[i][j] = edgeIndex
                cNodesTemp[int(onesPos[j])].append(edgeIndex)
                edgeIndex += 1
        vNodes = tf.constant(vNodesTemp, dtype=tf.int64)
        cNodes = tf.constant(cNodesTemp, dtype=tf.int64)
        vDegrees = np.repeat(vNodes.shape[1], repeats = vNodes.shape[0])
        cDegrees = np.repeat(cNodes.shape[1], repeats = cNodes.shape[0])
        EdgesCount = Edges.shape[0]
    else:
        code = load_code(path, '')
        H = code.H
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
        EdgesCount = num_edges

    CD = CodeDescriptor()
    CD.EdgesCount = EdgesCount
    CD.vNodes = vNodes
    CD.cNodes = cNodes
    CD.vDegrees = vDegrees
    CD.cDegrees = cDegrees
    return CD

def ReadLLRsTF(f, N):
    LLRsTemp = ReadLLRs(f,N)
    return tf.constant(LLRsTemp, dtype=tf.float64)
