### Append stop word ###
STOP_WORD = True
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

### Inference Types ###
GREEDY = 0
BEAM_1 = 1; BEAM_2 = 2; BEAM_3 = 3
VITERBI2 = 4; VITERBI3 = 5; VITERBI4 = 6
BEAM_K = 10
GREEDY_BI = 0
INFERENCE = VITERBI3

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
