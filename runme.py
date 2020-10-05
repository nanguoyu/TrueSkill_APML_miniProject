from QWD import Q9, Q56
from QWRY import Q4_Gibbs, Q4_plot
from QZWH import Q8, Q10
import numpy as np
if __name__ == '__main__':
    
    # In Question 4 we need first do Gibbs sampling with K
    K = 500  # tune the number of samples
    S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T = Q4_Gibbs(K, 1)
    np.save('./data/s1', S1[150:])
    np.save('./data/s2', S2[150:])
    # Then plot and save figures with suitble burn-in
    burnInNum = 120  # tune the number of burn-in
    Q4_plot(burnInNum, S1, S2, E_S1, E_S2, Var_S1, Var_S2, K)

    rank1 = Q56()
    Q8()
    Q9()
    Q10(rank1)

