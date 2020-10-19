"""
@File : QWRY.py
@Author: Ruiyun Wang
@Date : 2020/9/29
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time


def BayesianLinearRegression(mu_s1, mu_s2, sigma_s1, sigma_s2, sigma_t, t):
    S0 = np.mat([[sigma_s1 ** 2, 0], [0, sigma_s2 ** 2]])
    B = 1 / sigma_t ** 2
    X = np.mat([1, -1])
    m0 = np.mat([mu_s1, mu_s2]).T
    y = t
    SN_inverse = np.linalg.inv(S0) + B * X.T * X
    SN = np.linalg.inv(SN_inverse)
    m = SN * (np.linalg.inv(S0) * m0 + B * X.T * y)
    return SN, m


def Gibbs(mu_s1, mu_s2, sigma_s1, sigma_s2, sigma_t, y=1, K=100):
    # Gibbs sampling
    start_time = time.time()
    T = np.zeros(K)
    S1 = np.zeros(K)
    S2 = np.zeros(K)

    S1[0] = mu_s1
    S2[0] = mu_s2

    for i in range(K - 1):
        if y == 1:
            T[i + 1] = stats.truncnorm.rvs(a=0, b=np.inf, loc=S1[i] - S2[i], scale=sigma_t, size=1)
        elif y == -1:
            T[i + 1] = stats.truncnorm.rvs(a=-np.inf, b=0, loc=S1[i] - S2[i], scale=sigma_t, size=1)

        SN, m = BayesianLinearRegression(mu_s1, mu_s2, sigma_s1, sigma_s2, sigma_t, T[i + 1])
        S1[i + 1], S2[i + 1] = stats.multivariate_normal.rvs(mean=np.asarray(m).squeeze(), cov=SN, size=1)

    E_S1 = np.zeros(K)
    E_S2 = np.zeros(K)
    Var_S1 = np.zeros(K)
    Var_S2 = np.zeros(K)
    Var_T = np.zeros(K)

    for i in range(1, K):
        E_S1[i] = np.mean(S1[:i])
        Var_S1[i] = np.var(S1[:i])
        E_S2[i] = np.mean(S2[:i])
        Var_S2[i] = np.var(S1[:i])
        Var_T[i] = np.var(T[:i])

    cost_time = time.time() - start_time
    print("--- %s seconds ---" % cost_time)
    return S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T


def recoveryGaussian(mu1, mu2, sigma1, sigma2, sigmat, t):
    # Answer to Question 4.2, return is a multivariate normal distribution
    SN, m = BayesianLinearRegression(mu1, mu2, sigma1, sigma2, sigmat, t)
    ga = stats.multivariate_normal(mean=np.asarray(m).squeeze(), cov=SN)
    return ga


def burnIn(burnInNum, S1, S2, E_S1, E_S2, mu=25, K=100):
    # Plot the samples of the posterior distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.plot(S1)
    ax1.plot(S2)
    ax1.vlines(burnInNum, 18, 32)
    ax1.legend(['p(s1|y=1)', 'p(s2|y=1)', 'burn in'], loc='upper left', prop={'size': 9})
    # Plot the estimated means of the posterior distributions
    ax2.plot(E_S1)
    ax2.plot(E_S2)
    ax2.vlines(burnInNum, 0, 28)
    ax2.hlines(mu, 0, K - 1, 'g')
    ax2.legend(['Estimated mean of S1', 'Estimated mean of S2', 'True mean of S1 & S2'], loc='lower right',
               prop={'size': 9})
    fig.savefig('burnIn.png')


def histo(burnInNum, S1, S2, E_S1, E_S2, Var_S1, Var_S2, K):
    e_s1, e_s2, v_s1, v_s2 = np.mean(E_S1[burnInNum:]), np.mean(E_S2[burnInNum:]), np.mean(Var_S1[burnInNum:]), np.mean(
        Var_S2[burnInNum:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    # Plot the histogram for s1
    ax1.hist(S1[burnInNum:], bins=20, density=True)
    # Plot the approximated Gaussian posteriors for s1
    mu = e_s1
    sigma = v_s1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, K - burnInNum)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma))
    ax1.legend(['hist after burn in', 'approximated posterior'], loc='upper right', prop={'size': 9})
    # Plot the histogram for s2
    ax2.hist(S2[burnInNum:], bins=20, density=True)
    # Plot the approximated Gaussian posteriors for s2
    mu = e_s2
    sigma = v_s2
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, K - burnInNum)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma))
    ax2.legend(['hist after burn in', 'approximated posterior'], loc='upper right', prop={'size': 9})
    fig.savefig('histo.png')


def prior_n_posterior(burnInNum, E_S1, E_S2, Var_S1, Var_S2, mu=25, sigma=8.3, K=100):
    e_s1, e_s2, v_s1, v_s2 = np.mean(E_S1[burnInNum:]), np.mean(E_S2[burnInNum:]), np.mean(Var_S1[burnInNum:]), np.mean(
        Var_S2[burnInNum:])
    # Plot s1 prior
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, K)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma))
    # Plot s1 posterior
    x = np.linspace(e_s1 - 3 * v_s1, e_s1 + 3 * v_s1, K)
    ax1.plot(x, stats.norm.pdf(x, e_s1, v_s1))
    ax1.legend(['p(s1)', 'Approximated p(s1|y=1)'], loc='upper right')
    # Plot s2 prior
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, K)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma))
    # Plot s2 posterior
    x = np.linspace(e_s2 - 3 * v_s2, e_s2 + 3 * v_s2, K)
    ax2.plot(x, stats.norm.pdf(x, e_s2, v_s2))
    ax2.legend(['p(s2)', 'Approximated p(s2|y=1)'], loc='upper right')
    fig.savefig('pnp.png')


def Q4_Gibbs(K, sigma_t=3.3):
    # s1 ~ N(s1; mu, sigma)
    # s2 ~ N(s2; mu, sigma)
    # t ~ N(t; s1-s2, sigma_t)
    # y = sign(t)
    mu = 25
    sigma = 8.3

    S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T = Gibbs(mu, mu, sigma, sigma, sigma_t, 1, K)
    return S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T


def Q4_plot(burnInNum, S1, S2, E_S1, E_S2, Var_S1, Var_S2, K):
    burnIn(burnInNum, S1, S2, E_S1, E_S2, K)
    histo(burnInNum, S1, S2, E_S1, E_S2, Var_S1, Var_S2, K)
    prior_n_posterior(burnInNum, E_S1, E_S2, Var_S1, Var_S2, K)

# if __name__ == '__main__':
#     K = 500  # tune the number of samples
#     S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T = Q4_Gibbs(K, 1)

#     burnInNum = 120  # tune the number of burn-in
#     Q4_plot(burnInNum, S1, S2, E_S1, E_S2, Var_S1, Var_S2, K)
