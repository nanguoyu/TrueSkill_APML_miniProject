from QWD import Q9, Q56
from QWRY import Q4_Gibbs, Q4_plot

if __name__ == '__main__':
    
    # In Question 4 we need first do Gibbs sampling with K
    K = 500  # tune the number of samples
    S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T = Q4_Gibbs(K, 1)
    # Then plot and save figures with suitble burn-in
    burnInNum = 120  # tune the number of burn-in
    Q4_plot(burnInNum, S1, S2, E_S1, E_S2, Var_S1, Var_S2)
    
    Q56()
    Q9()
