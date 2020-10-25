"""
@File : QWRY.py
@Author: Ruiyun Wang
@Date : 2020/9/29
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time


def BayesianLinearRegression(mu_s1, mu_s2, sigma_1, sigma_2, sigma_3, t):
    S0 = np.mat([[sigma_1 ** 2, 0], [0, sigma_2 ** 2]])
    B = 1 / sigma_3 ** 2
    X = np.mat([1, -1])
    m0 = np.mat([mu_s1, mu_s2]).T
    y = t
    SN_inverse = np.linalg.inv(S0) + B * X.T * X
    SN = np.linalg.inv(SN_inverse)
    m = SN * (np.linalg.inv(S0) * m0 + B * X.T * y)
    return SN, m


def Gibbs(mu_s1, mu_s2, sigma_1, sigma_2, sigma_3, K=5000, y=1):
    # Gibbs Sampling
    start_time = time.time()
    T = np.zeros(K)
    S1 = np.zeros(K)
    S2 = np.zeros(K)

    S1[0] = mu_s1
    S2[0] = mu_s2

    for i in range(K - 1):
        if y == 1:
            T[i + 1] = stats.truncnorm.rvs(a=0, b=np.inf, loc=S1[i] - S2[i], scale=sigma_3, size=1)
        elif y == -1:
            T[i + 1] = stats.truncnorm.rvs(a=-np.inf, b=0, loc=S1[i] - S2[i], scale=sigma_3, size=1)

        SN, m = BayesianLinearRegression(mu_s1, mu_s2, sigma_1, sigma_2, sigma_3, T[i + 1])
        S1[i + 1], S2[i + 1] = stats.multivariate_normal.rvs(mean=np.asarray(m).squeeze(), cov=SN, size=1)

    E_S1 = np.zeros(K)
    E_S2 = np.zeros(K)
    # Var_S1 = np.zeros(K)
    # Var_S2 = np.zeros(K)
    # Var_T = np.zeros(K)
    for i in range(1, K):
        E_S1[i] = np.mean(S1[:i])
        # Var_S1[i] = np.var(S1[:i])
        E_S2[i] = np.mean(S2[:i])
        # Var_S2[i] = np.var(S1[:i])
        # Var_T[i] = np.var(T[:i])

    cost_time = time.time() - start_time
    print("--- %s seconds ---" % cost_time)
    return S1, S2, T, E_S1, E_S2


def burnIn(burnInNum, S1, S2, E_S1, E_S2, K, mu=25):
    # Plot the samples of the posterior distributions
    plt.figure(0)
    plt.plot(S1)
    plt.plot(S2)
    plt.vlines(burnInNum, -20, 70)
    plt.legend(['posterior s1', 'posterior s2', 'burn in'], loc='lower right', prop={'size': 9})
    plt.savefig('samples.png')

    plt.figure(1)
    # Plot the estimated means of the posterior distributions
    plt.plot(E_S1)
    plt.plot(E_S2)
    plt.hlines(mu, 0, K - 1, 'g')
    plt.vlines(burnInNum, 0, 45)
    plt.legend(['Estimated mean of S1', 'Estimated mean of S2', 'Prior mean of S1 & S2', 'previous burn in'],
               loc="lower right", prop={'size': 9})
    plt.savefig('burnIn.png')


def approximation(burnInNum, S1, S2, K):
    # Answer to Question 4.2, returning a Gaussian approximation of the posterior distribution of the skills
    m_s1, m_s2, v_s1, v_s2 = np.mean(S1[burnInNum:]), np.mean(S2[burnInNum:]), np.var(S1[burnInNum:]), np.var(
        S2[burnInNum:])

    x1 = np.linspace(m_s1 - 3 * np.sqrt(v_s1), m_s1 + 3 * np.sqrt(v_s1), K - burnInNum)
    x2 = np.linspace(m_s2 - 3 * np.sqrt(v_s2), m_s2 + 3 * np.sqrt(v_s2), K - burnInNum)

    s1_post = stats.norm.pdf(x1, m_s1, np.sqrt(v_s1))
    s2_post = stats.norm.pdf(x2, m_s2, np.sqrt(v_s2))
    return x1, x2, s1_post, s2_post


def histo(burnInNum, S1, S2, K):
    plt.figure(2, figsize=(4.5, 6))
    x1, x2, s1_post, s2_post = approximation(burnInNum, S1, S2, K)
    # Plot the histogram for s1 and s2
    plt.hist(S1[burnInNum:], bins=30, density=True, alpha=0.5)
    plt.hist(S2[burnInNum:], bins=30, density=True, alpha=0.5)
    # Plot the approximated Gaussian posteriors for s1 and s2
    plt.plot(x1, s1_post, color=[0.39, 0.59, 0.80])
    plt.plot(x2, s2_post, 'orange')
    plt.legend(['fitted s1', 'fitted s2', 'histogram of s1', 'histogram of s2'], loc='upper right', prop={'size': 9})
    plt.savefig('histo.png')


def prior_n_posterior(burnInNum, S1, S2, K, mu=25, sigma=8.3):
    x1, x2, s1_post, s2_post = approximation(burnInNum, S1, S2, K)
    # Plot s1 prior and s2 prior
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, K - burnInNum)
    plt.figure(3)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r')
    # Plot the approximated Gaussian posteriors for s1 and s2
    plt.plot(x1, s1_post, color=[0.39, 0.59, 0.80])
    plt.plot(x2, s2_post, 'orange')
    plt.legend(['prior s1 and s2', 'posterior s1', 'posterior s2'], loc='upper right', prop={'size': 9})
    plt.savefig('pnp.png')


def Q4_Gibbs(K, sigma_3=3.3**2, y=1):
    # s1 ~ N(s1; mu, sigma)
    # s2 ~ N(s2; mu, sigma)
    # t ~ N(t; s1-s2, sigma_3)
    # y = sign(t)
    mu = 25
    sigma = 8.3
    S1, S2, T, E_S1, E_S2 = Gibbs(mu, mu, sigma, sigma, sigma_3, K, y)
    return S1, S2, T, E_S1, E_S2


def Q4_plot(burnInNum, S1, S2, E_S1, E_S2, K):
    burnIn(burnInNum, S1, S2, E_S1, E_S2, K)
    histo(burnInNum, S1, S2, K)
    prior_n_posterior(burnInNum, S1, S2, K)


# if __name__ == '__main__':
#     K = 5000  # number of samples
#     S1, S2, T, E_S1, E_S2 = Q4_Gibbs(K)
# 
#     burnInNum = 2200  # tune the number of burn-in
#     Q4_plot(burnInNum, S1, S2, E_S1, E_S2, K)
