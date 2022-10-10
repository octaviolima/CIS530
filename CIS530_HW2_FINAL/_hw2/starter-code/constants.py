### Append stop word ###
STOP_WORD = True
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

### Inference Types ###
GREEDY = 0
BEAM_4 = 1; BEAM_2 = 2; BEAM_3 = 3
VITERBI2 = 4; VITERBI3 = 5; VITERBI4 = 6
GREEDY_BI = 0
GREEDY_TRI = 1
GREEDY_QUAD = 2

# !!!!!!!! Main constants to adjust when runnning the code !!!!!!!!
BEAM_K = 3
GREEDY_TYPE = GREEDY_QUAD
INFERENCE = VITERBI2

### Smoothing Types ###
LAPLACE = 0; LAPLACE_FACTOR = .01
INTERPOLATION = 1; LAMBDAS =  [0,.90199,.098009]
SMOOTHING = LAPLACE

### Append stop word ###
STOP_WORD = True

### Capitalization
CAPITALIZATION = True

# NGRAMM
NGRAMM = 3

## Handle unknown words TnT style
TNT_UNK = True
UNK_C = 10 #words with count to be considered
UNK_M = 10 #substring length to be considered



# NOTE: VITERBI4 takes a while to run