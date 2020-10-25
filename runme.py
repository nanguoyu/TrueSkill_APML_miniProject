from QWD import Q9, Q56
from QWRY import Q4_Gibbs, Q4_plot
from QZWH import Q8, Q10
import numpy as np
import pandas as pd
if __name__ == '__main__':
    
    # In Question 4 we need first do Gibbs sampling with K
    K = 5000  # number of samples
    S1, S2, T, E_S1, E_S2 = Q4_Gibbs(K)
    
    # Then plot and save figures with suitble burn-in
    burnInNum = 2200  # tune the number of burn-in
    Q4_plot(burnInNum, S1, S2, E_S1, E_S2, K)
    
    np.save('./data/s1', S1[burnInNum:])
    np.save('./data/s2', S2[burnInNum:])

    rank1 = Q56()
    Q8()
    Q9()
    rank1.to_csv('./data/rank1.csv')
    Q10(pd.read_csv('./data/rank1.csv'))

