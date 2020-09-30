"""
@File : QWD.py
@Author: Dong Wang
@Date : 2020/9/29
"""

# This doc include the solution of Q5, Q6, and Q9

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import pandas as pd
from numpy import array

# hyperparameters

mu_s1 = 25
mu_s2 = 25
segma_s1 = 8.3
segma_s2 = 8.3
segma_t = 0.3
K = 500
burnInNum = 180


# HelpFunction for Gibbs
def BayesianLinearRegression(mu_s1, mu_s2, segma_s1, segma_s2, segma_t, t):
    S0 = np.mat([[segma_s1, 0], [0, segma_s2]])
    B = segma_t
    X = np.mat([1, -1])
    m0 = np.mat([mu_s1, mu_s2]).T
    y = t
    SN_inverse = np.linalg.inv(S0) + B * X.T * X
    SN = np.linalg.inv(SN_inverse)
    m = SN * (np.linalg.inv(S0) * m0 + B * X.T * y)
    return SN, m


# Gibbs sampling
def Gibbs(mu_s1, mu_s2, segma_s1, segma_s2, segma_t, y=1):
    start_time = time.time()
    T = np.zeros(K)
    S1 = np.zeros(K)
    S2 = np.zeros(K)

    S1[0] = mu_s1
    S2[0] = mu_s2

    for i in range(K - 1):
        if y == 1:
            T[i + 1] = stats.truncnorm.rvs(a=0, b=np.inf, loc=S1[i] - S2[i], scale=segma_t, size=1)
        elif y == -1:
            T[i + 1] = stats.truncnorm.rvs(a=-np.inf, b=0, loc=S1[i] - S2[i], scale=segma_t, size=1)

        SN, m = BayesianLinearRegression(mu_s1, mu_s2, segma_s1, segma_s2, segma_t, T[i + 1])
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
    return S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T


# initial skills with pre-defined parameters
def buildSkills(teams):
    skills = pd.DataFrame(columns=["Team", "skill", "variance"])
    skills.Team = pd.Series(teams)
    skills.skill = pd.Series(np.ones(teams.shape[0]) * 25)
    skills.variance = pd.Series(np.ones(teams.shape[0]) * 8.3)
    return skills


# run each match for Q5 and Q6
def run(d, skills):
    for i in range(len(d)):
        t1, t2 = d.iloc[i, 2:4].values
        score1, score2 = d.iloc[i, 4:6].values
        if score1 == score2:
            continue
        s1 = skills.loc[skills.Team == t1, 'skill'].values[0]
        s2 = skills.loc[skills.Team == t2, 'skill'].values[0]
        v1 = skills.loc[skills.Team == t1, 'variance'].values[0]
        v2 = skills.loc[skills.Team == t2, 'variance'].values[0]

        S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T = Gibbs(mu_s1=s1, mu_s2=s2, segma_s1=v1, segma_s2=v2, segma_t=0.3,
                                                             y=np.sign(score1 - score2))
        es1, es2, em1, em2 = np.mean(E_S1[burnInNum:]), np.mean(E_S2[burnInNum:]), np.mean(Var_S1[burnInNum:]), np.mean(
            Var_S2[burnInNum:])
        skills.loc[skills.Team == t1, 'skill'] = es1
        skills.loc[skills.Team == t2, 'skill'] = es2
        skills.loc[skills.Team == t1, 'variance'] = em1
        skills.loc[skills.Team == t2, 'variance'] = em2
    result = sorted(skills.values, key=lambda s: s[1] - 3 * s[2], reverse=True)
    # print(result)
    return result


# run each match for Q9
def run2(d, skills):
    for i in range(len(d)):
        t1, t2 = d.iloc[i, [1, 3]].values
        score1, score2 = d.iloc[i, [2, 4]].values
        if score1 == score2:
            continue
        s1 = skills.loc[skills.Team == t1, 'skill'].values[0]
        s2 = skills.loc[skills.Team == t2, 'skill'].values[0]
        v1 = skills.loc[skills.Team == t1, 'variance'].values[0]
        v2 = skills.loc[skills.Team == t2, 'variance'].values[0]

        S1, S2, T, E_S1, E_S2, Var_S1, Var_S2, Var_T = Gibbs(mu_s1=s1, mu_s2=s2, segma_s1=v1, segma_s2=v2, segma_t=0.3,
                                                             y=np.sign(score1 - score2))
        es1, es2, em1, em2 = np.mean(E_S1[burnInNum:]), np.mean(E_S2[burnInNum:]), np.mean(Var_S1[burnInNum:]), np.mean(
            Var_S2[burnInNum:])
        skills.loc[skills.Team == t1, 'skill'] = es1
        skills.loc[skills.Team == t2, 'skill'] = es2
        skills.loc[skills.Team == t1, 'variance'] = em1
        skills.loc[skills.Team == t2, 'variance'] = em2
    result = sorted(skills.values, key=lambda s: s[1] - 3 * s[2], reverse=True)
    # print(result)
    return result


def Q56():
    print("Solving Q5")
    path = "./data/"
    filename = "SerieA.csv"
    df = pd.read_csv(path + filename)
    teams = np.unique(df.loc[:, ['team1', 'team2']].values.reshape(-1))
    print("There are totally %s teams in SerieA.csv" % teams.shape[0])
    sortedSkills = run(d=df, skills=buildSkills(teams))
    sortedSkills2 = run(df.sample(frac=1), buildSkills(teams))
    rank1 = pd.DataFrame(sortedSkills, columns=["Team", "skill", "variance"])
    rank2 = pd.DataFrame(sortedSkills2, columns=["Team", "skill", "variance"])
    rank1.to_csv("Q5Rank.csv")
    print("Q5:The rank result is saved in Q5Rank.csv")

    # Predict and update step by step
    d = df
    p = 0
    for i in range(len(d)):
        t1, t2 = d.iloc[i, 2:4].values
        score1, score2 = d.iloc[i, 4:6].values
        if score1 == score2:
            continue
        s1 = rank1.loc[rank1.Team == t1, 'skill'].values[0]
        s2 = rank1.loc[rank1.Team == t2, 'skill'].values[0]
        v1 = rank1.loc[rank1.Team == t1, 'variance'].values[0]
        v2 = rank1.loc[rank1.Team == t2, 'variance'].values[0]
        print(s1, s2, v1, v2)
        m = np.array([s1, s2])
        SN = np.array([[v1, 0], [0, v2]])
        ga = stats.multivariate_normal(mean=m, cov=SN)
        S = ga.rvs(1000)
        r_pred = np.sign(np.mean(np.sign(S[:, 0] - S[:, 1])))
        r_true = np.sign(score1 - score2)
        if r_pred == 1:
            if r_true == 1:
                rank1.loc[rank1.Team == t1, 'skill'] += 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] -= 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
                p += 1
            elif r_true == -1:
                rank1.loc[rank1.Team == t1, 'skill'] -= 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] += 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
        elif r_pred == -1:
            if r_true == 1:
                rank1.loc[rank1.Team == t1, 'skill'] += 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] -= 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
            elif r_true == -1:
                rank1.loc[rank1.Team == t1, 'skill'] -= 0.005 * abs(s1 - s2) * v1
                rank1.loc[rank1.Team == t2, 'skill'] += 0.005 * abs(s1 - s2) * v2
                rank1.loc[rank1.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank1.loc[rank1.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
                p += 1
    print("Solving Q6")
    print("Q6:The accuracy of predicting SeriesA.csv", 100 * p / df.shape[0], "%")
    return rank1


def Q9():
    print("Solving Q9")
    path = "./data/"
    ExtraDataFile = "hockey.csv"
    hockey = pd.read_csv(path + ExtraDataFile, usecols=[0, 1, 2, 3, 4])
    hockey.columns.values[2] = "G1"
    hockey.columns.values[4] = "G2"
    hockeyTeams = np.unique(hockey.loc[:, ['Visitor', 'Home']].values.reshape(-1))
    print("There are totally %s hockey teams" % hockeyTeams.shape[0])
    sortedHockeySikll = run2(d=hockey, skills=buildSkills(hockeyTeams))
    rank3 = pd.DataFrame(sortedHockeySikll, columns=["Team", "skill", "variance"])

    d = hockey
    p = 0
    for i in range(len(d)):
        t1, t2 = d.iloc[i, [1, 3]].values
        score1, score2 = d.iloc[i, [2, 4]].values
        if score1 == score2:
            continue
        s1 = rank3.loc[rank3.Team == t1, 'skill'].values[0]
        s2 = rank3.loc[rank3.Team == t2, 'skill'].values[0]
        v1 = rank3.loc[rank3.Team == t1, 'variance'].values[0]
        v2 = rank3.loc[rank3.Team == t2, 'variance'].values[0]
        m = np.array([s1, s2])
        SN = np.array([[v1, 0], [0, v2]])
        ga = stats.multivariate_normal(mean=m, cov=SN)
        S = ga.rvs(1000)
        r_pred = np.sign(np.mean(np.sign(S[:, 0] - S[:, 1])))
        r_true = np.sign(score1 - score2)
        if r_pred == 1:
            if r_true == 1:
                rank3.loc[rank3.Team == t1, 'skill'] += 0.005 * abs(s1 - s2) * v1
                rank3.loc[rank3.Team == t2, 'skill'] -= 0.005 * abs(s1 - s2) * v2
                rank3.loc[rank3.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank3.loc[rank3.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
                p += 1
            elif r_true == -1:
                rank3.loc[rank3.Team == t1, 'skill'] -= 0.005 * abs(s1 - s2) * v1
                rank3.loc[rank3.Team == t2, 'skill'] += 0.005 * abs(s1 - s2) * v2
                rank3.loc[rank3.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank3.loc[rank3.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
        elif r_pred == -1:
            if r_true == 1:
                rank3.loc[rank3.Team == t1, 'skill'] += 0.005 * abs(s1 - s2) * v1
                rank3.loc[rank3.Team == t2, 'skill'] -= 0.005 * abs(s1 - s2) * v2
                rank3.loc[rank3.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank3.loc[rank3.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
            elif r_true == -1:
                rank3.loc[rank3.Team == t1, 'skill'] -= 0.005 * abs(s1 - s2) * v1
                rank3.loc[rank3.Team == t2, 'skill'] += 0.005 * abs(s1 - s2) * v2
                rank3.loc[rank3.Team == t1, 'variance'] *= 1 - 0.005 * v1 / abs(s1 - s2)
                rank3.loc[rank3.Team == t2, 'variance'] *= 1 - 0.005 * v2 / abs(s1 - s2)
                p += 1
    print("Q9:The accuracy of predicting hockey.csv", 100 * p / d.shape[0], "%")

