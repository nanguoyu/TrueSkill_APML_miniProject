"""
@File: QZWH.py
@Author: Wenhao Zhu
@Date: 29.09.2020
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import truncnorm
from scipy.stats import norm
from matplotlib import pyplot as plt
import QWD as wd
from QWRY import histo, approximation
import pandas as pd


def multiply_gaussian(mean0, var0, mean1, var1):
    # computes the Gaussian distribution N(m,s) being proportional to N(m1,s1)*N(m2,s2)
    #
    # Input:
    # mean0, var0: mean and variance of first Gaussian
    # mean1, var1: mean and variance of second Gaussian
    #
    # Output:
    # mean, var: mean and variance of the product Gaussian
    # t = mean0 + mean1
    var = 1 / (1 / var0 + 1 / var1)
    mean = (mean0 / var0 + mean1 / var1) * var
    return mean, var


def divide_gaussian(mean0, var0, mean1, var1):
    # computes the Gaussian distribution N(m,s) being proportional to N(m1,s1)/N(m2,s2)
    #
    # Input:
    # mean0, var0: mean and variance of the numerator Gaussian
    # mean1, var1: mean and variance of the denominator Gaussian
    #
    # Output:
    # mean, var: mean and variance of the quotient Gaussian
    mean, var = multiply_gaussian(mean0, var0, mean1, -var1)
    return mean, var


def truncated_gaussian(a, b, mean0, var0):
    # computes the mean and variance of a truncated Gaussian distribution
    #
    # Input:
    # a, b: The interval [a, b] on which the Gaussian is being truncated
    # mean0, var0: mean and variance of the Gaussian which is to be truncated
    #
    # Output:
    # mean, var: mean and variance of the truncated Gaussian
    # scale interval with mean and variance
    a_scaled, b_scaled = (a - mean0) / np.sqrt(var0), (b - mean0) / np.sqrt(var0)
    mean = truncnorm.mean(a_scaled, b_scaled, loc=mean0, scale=np.sqrt(var0))
    var = truncnorm.var(a_scaled, b_scaled, loc=mean0, scale=np.sqrt(var0))
    return mean, var


def Q8():
    # means and variances for f(s1), f(s2) and f(t)
    mean_s1 = 25
    var_s1 = 8.3**2
    mean_s2 = 25
    var_s2 = 8.3**2
    var_t = 3.3**2
    y = 1

    # Message mu3 from f(s1) to node s1
    mu3_mean = mean_s1
    mu3_var = var_s1

    # Message mu4 from node s1 to factor f(t)
    mu4_mean = mu3_mean
    mu4_var = mu3_var

    # Message mu5 from f(s2) to node s2
    mu5_mean = mean_s2
    mu5_var = var_s2

    # Message mu6 from node s2 to factor f(t)
    mu6_mean = mu5_mean
    mu6_var = mu5_var

    # Message mu7 from factor f(t) to node t
    mu7_mean = mu4_mean - mu6_mean
    mu7_var = var_s1 + var_s2 + var_t

    # Do moment matching of the marginal of t
    if y == 1:
        a, b = 0, np.inf
    else:
        a, b = -np.inf, 0

    pt_mean, pt_var = truncated_gaussian(a, b, mu7_mean, mu7_var)

    # Compute the message from node t to factor f(t)
    mu8_mean, mu8_var = divide_gaussian(pt_mean, pt_var, mu7_mean, mu7_var)

    # Message mu9 from factor f(t) to node s1
    mu9_mean = mu6_mean + mu8_mean
    mu9_var = var_t + mu8_var + var_s2

    # Message mu10 from factor f(t) to node s2
    mu10_mean = mu4_mean - mu8_mean
    mu10_var = var_t + mu8_var + var_s1

    # Compute the marginal of s1
    p_s1_mean, p_s1_var = multiply_gaussian(mu3_mean, mu3_var, mu9_mean, mu9_var)
    p_s2_mean, p_s2_var = multiply_gaussian(mu5_mean, mu5_var, mu10_mean, mu10_var)

    L = 100  # number of samples
    x = np.linspace(mean_s1 - 3 * np.sqrt(var_s1), mean_s1 + 3 * np.sqrt(var_s1), 100)
    y = np.linspace(mean_s2 - 3 * np.sqrt(var_s2), mean_s2 + 3 * np.sqrt(var_s2), 100)

    s1_pdf = norm.pdf(x, p_s1_mean, np.sqrt(p_s1_var))
    s2_pdf = norm.pdf(y, p_s2_mean, np.sqrt(p_s2_var))

    S1 = np.load('./data/s1.npy')
    S2 = np.load('./data/s2.npy')

    burnInNum = 2200
    K = 5000

    m_s1, m_s2, v_s1, v_s2 = np.mean(S1[burnInNum:]), np.mean(S2[burnInNum:]), np.var(S1[burnInNum:]), np.var(
        S2[burnInNum:])
    x1 = np.linspace(m_s1 - 3 * np.sqrt(v_s1), m_s1 + 3 * np.sqrt(v_s1), K - burnInNum)
    x2 = np.linspace(m_s2 - 3 * np.sqrt(v_s2), m_s2 + 3 * np.sqrt(v_s2), K - burnInNum)

    s1_post = stats.norm.pdf(x1, m_s1, np.sqrt(v_s1))
    s2_post = stats.norm.pdf(x2, m_s2, np.sqrt(v_s2))

    plt.figure(8)

    # Plot the histogram for s1 and s2
    plt.hist(S1[burnInNum:], bins=30, density=True, alpha=0.5)
    plt.hist(S2[burnInNum:], bins=30, density=True, alpha=0.5)

    # Plot the message passing distribution
    plt.plot(x, s1_pdf, linewidth=2, label="s1 message passing")
    plt.plot(y, s2_pdf, linewidth=2, label="s2 message passing")
    # Plot the approximated Gaussian posteriors for s1 and s2
    plt.plot(x1, s1_post, color=[0.39, 0.59, 0.80])
    plt.plot(x2, s2_post, 'orange')

    plt.legend(['s1 message passing', 's2 message passing','fitted s1', 'fitted s2', 'histogram of s1', 'histogram of s2'], loc='upper right', prop={'size': 9})
    plt.savefig('Q8.png')

    # plt.close()
    # plt.figure()
    # plt.plot(x1, s1_post, color=[0.39, 0.59, 0.80])
    # plt.plot(x2, s2_post, 'orange')
    # plt.hist(S1, label="s1, Gibbs Sampling", bins=50, density=True)
    # plt.title("s1 message passing")
    # plt.plot(x, s1_pdf, linewidth=2, label="s1 message passing")
    # plt.legend()
    # plt.savefig('s1message.png')
    #
    # plt.close()
    # plt.figure()
    # plt.hist(S2, label="s2, Gibbs Sampling", bins=50, density=True)
    # plt.title("s2 message passing")
    # plt.plot(x, s2_pdf, linewidth=2, label="s2 message passing")
    # plt.legend()
    # plt.savefig('s2message.png')


def Q10(rank1):
    # momentum methods insert into prediction method
    print("Solving Q10")
    path = "./data/"
    filename = "SerieA.csv"
    df = pd.read_csv(path + filename)
    lastUpdates = {}
    i = 0
    for name in rank1.Team.values:
        lastUpdates[name] = rank1.skill.values[i]
        i = i + 1
    d = df
    p = 0
    numOfDraw = 0
    eta = 0.05
    for i in range(len(d)):
        if i == 0:
            eta = 0
        else:
            eta = 0.8
        t1, t2 = d.iloc[i, 2:4].values
        score1, score2 = d.iloc[i, 4:6].values
        if score1 == score2:
            numOfDraw = numOfDraw+1
            continue
        s1 = rank1.loc[rank1.Team == t1, 'skill'].values[0]
        s2 = rank1.loc[rank1.Team == t2, 'skill'].values[0]
        v1 = rank1.loc[rank1.Team == t1, 'variance'].values[0]
        v2 = rank1.loc[rank1.Team == t2, 'variance'].values[0]
        r_pred = 1 if (s1-3*v1) > (s2-3*v2) else -1
        r_true = np.sign(score1 - score2)
        if r_pred == 1:
            if r_true == 1:
                rank1.loc[rank1.Team == t1, 'skill'] += 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] -= 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'skill'] += eta * lastUpdates[t1]
                rank1.loc[rank1.Team == t2, 'skill'] += eta * lastUpdates[t2]
                lastUpdates[t1] += 0.005 * abs(s1 - s2) * v1
                lastUpdates[t2] -= 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
                p += 1
            elif r_true == -1:
                rank1.loc[rank1.Team == t1, 'skill'] -= 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] += 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'skill'] += eta * lastUpdates[t1]
                rank1.loc[rank1.Team == t2, 'skill'] += eta * lastUpdates[t2]
                lastUpdates[t1] -= 0.005 * abs(s1 - s2) * v1
                lastUpdates[t2] += 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
        elif r_pred == -1:
            if r_true == 1:
                rank1.loc[rank1.Team == t1, 'skill'] += 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] -= 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'skill'] += eta * lastUpdates[t1]
                rank1.loc[rank1.Team == t2, 'skill'] += eta * lastUpdates[t2]
                lastUpdates[t1] += 0.005 * abs(s1 - s2) * v1
                lastUpdates[t2] -= 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
            elif r_true == -1:
                rank1.loc[rank1.Team == t1, 'skill'] -= 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] += 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'skill'] += eta * lastUpdates[t1]
                rank1.loc[rank1.Team == t2, 'skill'] += eta * lastUpdates[t2]
                lastUpdates[t1] -= 0.005 * abs(s1 - s2) * v1
                lastUpdates[t2] += 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
                p += 1

    print("Q10:The accuracy of predicting hockey.csv", 100 * p / (df.shape[0]-numOfDraw), "%")
