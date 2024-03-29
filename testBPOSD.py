# import numpy as np
# from ldpc.codes import rep_code
# from ldpc.bp_decode_sim import classical_decode_sim
# from bposd.css import css_code

# d=500

# hz = np.loadtxt("codesQLDPC\Hz_400_16.txt")
# pcm = hz
# #pcm=np.loadtxt("codesQLDPC\H_400_16.txt") # ("codesHDPC\BCH_15_11.txt")#
# error_rate=0.09049228825585272

# output_dict={}
# output_dict['code_type']=f"rep_code_{d}"

# output_dict=classical_decode_sim(
#     pcm,
#     error_rate,
#     target_runs=10000,
#     max_iter=400,
#     seed=100,
#     bp_method='ms',
#     ms_scaling_factor=0,
#     output_file="classical_bp_decode_sim_output.json",
#     output_dict=output_dict
# )

# print(output_dict)


# from bposd.css import css_code

# hx = np.loadtxt("codesQLDPC\Hx.txt")
# hz = np.loadtxt("codesQLDPC\Hz.txt")
# qcode = css_code(hx, hz)
# lx = qcode.lx
# lz = qcode.lz

# np.savetxt("codesQLDPC\Lx.txt", lx, fmt='%d', delimiter=" ")
# np.savetxt("codesQLDPC\Lz.txt", lz, fmt='%d', delimiter=" ")

# from bposd.css import bposd_decoder
# from ldpc.codes import rep_code
# from bposd.hgp import hgp

# h=rep_code(3)
# surface_code=hgp(h1=h,h2=h,compute_distance=True) #nb. set compute_distance=False for larger codes
# #surface_code.test()

# bpd=bposd_decoder(
#     pcm,#the parity check matrix
#     error_rate=error_rate,
#     channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
#     max_iter=400, #the maximum number of iterations for BP)
#     bp_method="Ps",
#     ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
#     osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
#     osd_order=7 #the osd search depth
#     )


import numpy as np
from bposd.hgp import hgp
from bposd.css_decode_sim import css_decode_sim
from bposd.css import css_code

# h=np.loadtxt("codesQLDPC/mkmn_16_4_6.txt").astype(int)
# qcode=hgp(h) # construct quantum LDPC code using the symmetric hypergraph product

hx = np.loadtxt("codesQLDPC\Hx_400_16.txt")
hz = np.loadtxt("codesQLDPC\Hz_400_16.txt")
qcode = css_code(hx, hz)

osd_options={
'error_rate': 0.05,
'target_runs': 1000,
'xyz_error_bias': [1, 0, 0],
'output_file': 'test.json',
'bp_method': "ps",
'ms_scaling_factor': 0,
'osd_method': "OSD_CS",
'osd_order': 60,
'channel_update': None,
'seed': 100,
'max_iter': 400,
'output_file': "test.json"
}

lk = css_decode_sim(hx=qcode.hx, hz=qcode.hz, **osd_options)
print(lk.bp_logical_error_rate)
waleed = 0