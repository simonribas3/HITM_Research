
# jse-main.py
# JSE four factor model

"""
Created on Mon Dec 6, 2021
Updated August 9, 2022
@author: Alec Kercheval

Four Factor code implementation: Simon Ribas
"""

'''
program to run simulations demonstrating JSE improvements for various metrics
Notation:  GPS = JSE in the simple case of shrinkage target at the vector of ones.
The SimulationGPS class contains the random variable calls that produce beta and
the returns matrix.
Here we have:
ComputePCA_GPS() in which the eigenvalues and eigenvectors are extracted from a supplied sample covariance matrix S.
ComputeMRPortfolio() implements the formula for the minimum risk portfolio depending on
input parameters beta, delta, sigma
The main loop iterates through NumExperiments to compute true and
estimated min risk portfolios, tracking error, and
variance forecast ratios.
Choices for comparison are:
raw and corrected evalues against PCA and JSE evectors
This gives 4 combinations to provide 4 min var portfolios for comparison.
We look at tracking error (comparing to true portfolio), VFR (estimated to true risk of estimated portfolio), and true variance ratios (ratio of true variance of true portfolio to true variance of estimated portfolio.)
Tr = (w_est - w)Sigma(w_est - w)
VFR = w_est Sigma_est w_est / w_est Sigma w_est
TVR = w Sigma w / w_est Sigma w_est
Output is boxplot figures summarizing NumExperiments saved to files in the current directory.
Variable notation and formulas come from GPS and MAPS papers.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jse_support as sjse
import scipy 
from scipy.optimize import minimize
from numpy import linalg as la

def ComputePCA_GPS(S, Srank, Sdim):
    evalues, evectors = la.eigh(S)
    h = [] # all our eigenvectors \beta, F1, F2, F3
    sp2 = [] # corresponding eigenvalues

    for i in range(1, 5): 
        h.append(evectors[:, -i])
        sp2.append(evalues[-i])

    lp2 = (np.sum(evalues ) -sp2[0] ) /(Srank -1)
    psi2 = (sp2[0] - lp2) / sp2[0]  # this is the psi^2 term from the GPS paper
    all_ones = np.ones(Sdim)
    q = all_ones / la.norm(all_ones) #north pole, unit vector

    hq = np.dot(h[0], q)  # inner product of h and q
    if hq < 0:
        h[0] = -h[0]  # choose e-vector h with positive mean
        hq = -hq
    elif hq == 0:
        print("error: h is orthogonal to q")

    tau = (1 - psi2) * hq / (psi2 - hq * hq)  # equation (21) from the dispersion bias
    h_shr = h[0] + tau * q  # h_GPS before normalizing

    h_shr_norm = (1/la.norm(h_shr))*h_shr #h_GPS normalized

    for i in range(1,4):  # now lets normalize the rest of the principal componenets
        h[i] = (1/la.norm(h[i]))*h[i]

    # up to here we have computed a list of our first four principal components normalized except for PCA, and their respective eigenvalues
    # h_shr_norm which is our corrected beta vector

    # now depending on the flag we will return two different things

    if FactorFlag == 0: 
        return h[0], h_shr_norm, sp2[0], lp2 # PCA principal comp, JSE prinpcal comp, leading eigenval, avg of rest eigenvals
    elif FactorFlag == 1: 
        return h, h_shr_norm, sp2 # Principal Components list, JSE princiapl comp, Corresponding eigenvalues. 
    else: 
        print('please correct flag')

###  end def


def ComputeMRPortfolio(
    h, **kwargs  # computes weights for one factor covariance matrix:
    ) -> list: 
    
    try: 
        try:
            sigma2 = kwargs['mvar']
        except:
            sigma2_l = kwargs['mvar_l']
    except:
        d2 = kwargs['svar']
        p_eta = kwargs['p_eta']
        delta2 = kwargs['delta2']

    all_ones = np.ones(MaxAssets)
    
    if FactorFlag == 0:
        # outputs w = argmin w^T Sigma w, subj to w^T e = 1.
        # Here Sigma is the real or estimated covariance matrix, depending on inputs
        # p = dimension of Sigma = number of assets
        # p_eta, delta2, h determine Sigma = p_eta hh^T + delta2 I
        # Notation follows MAPS, section 3

        q = all_ones / la.norm(all_ones)  # north pole, unit vector
        hq = np.dot(h, q) # hq is just the h vector in the direction of the north pole
        k2 = delta2 / p_eta
        rho = (1 + k2) / hq
        w = ((rho * q) - h) / ((rho - hq) * np.sqrt(p))

    elif FactorFlag == 1:

        Omega = sigma2*np.diag([1]*4) # this creates Omega diagonal (17)
        Delta = d2*np.diag([1]*MaxAssets) # this creates Delta diagonal in (17)
        cov_est = np.dot(np.dot(np.array(h).T, Omega), np.array(h)) + Delta # this is (17)
        w_num = np.dot(np.linalg.inv(cov_est),all_ones) # numerator of solving for w
        w = w_num/np.dot(all_ones.T,w_num) # final computation solving for w

    return w



### end def

#####################################################################
### main program   ##################################################
#####################################################################

# set up parameters for input to SimulationGPS object

DayString = "d220809"
MaxAssets = 500  # default 500
NumExperiments = 5  # default 400
NumPeriods = 252  # default 252

BetaMean = 1.0
BetaStDev = 0.5

Factor1StDev = 0.16 / np.sqrt(252)  # default 0.16/sqrt(252)
SpecificStDev = 0.6 / np.sqrt(252)  # daily vol from annual
Factor2StDev = .04 / np.sqrt(252)
Factor3StDev = .04 / np.sqrt(252)
Factor4StDev = .08 / np.sqrt(252)

FactorFlag = 1  # 0 for one factor; 1 for four factors
NormalFlag = 2  # 0 for Normal specific returns, 1 for double exponential, 2 for student's t

##########################################################################
################## CREATE Z MATRIX OF RESIDUAL RETURNS ###################
##########################################################################

def Compute_Zmatrix_cost(
    x, Bstar, Returns, 
    ) -> float:
    error = Returns - np.matmul(Bstar, x)
    returnVALUE = np.linalg.norm(error, 2)**2
    return returnVALUE

def getJSE_BSTAR(
    h, h_JSE
    ) -> list:
    H = h
    H[0] = h_JSE
    return H


def Compute_Zmatrix( # Function returns the JSE and PCA residuals matrix (keep at low experiments for quick tests)
    Returns: list
    ) -> list:

    numFactors = 4
    initialGuess = np.random.randn(numFactors)  # The initial guess for scipy optimizer
    Z = np.zeros((MaxAssets, NumPeriods, NumExperiments))  # The Residuals matrix
    Z_jse = np.zeros((MaxAssets, NumPeriods, NumExperiments))  # The Residuals matrix for JSE

    for t in range(NumExperiments): # Looping through all the experiments
        R_exper = Returns[:, :, t] 
        S = np.matmul(R_exper, R_exper.transpose()) / NumPeriods  # Create the Sample covariance matrix for experiment i
        h, h_JSE, sp2 = ComputePCA_GPS(S, NumPeriods, len(S), 1)

        Bstar = np.array(h).T  # This is the BSTAR in better betas, factor exposures
        Bstar_JSE = np.array(getJSE_BSTAR(h,h_JSE)).T  # This is the BSTAR for JSE

        psi_exp_i = np.zeros((4, NumPeriods)) 
        psi_exp_i_jse = np.zeros((4,NumPeriods))

        for j in range(NumPeriods):
            R_day = R_exper[:, j]

            psi_day = minimize(Compute_Zmatrix_cost,initialGuess,method = 'CG',       # Optimizer PCA Conjugate Gradient 
            args =(Bstar,np.reshape(R_day, (MaxAssets,))),
            options = {'maxiter':10000, 'maxfev':10000, 'xatol': 1e-8, 'disp': True})

            psi_day_jse = minimize(Compute_Zmatrix_cost,initialGuess,method = 'CG',    # Optimizer JSE Conjugate Gradient
            args =(Bstar_JSE,np.reshape(R_day, (MaxAssets,))),
            options = {'maxiter':10000, 'maxfev':10000, 'xatol': 1e-8, 'disp': True})

            
            psi_exp_i[:, j] = psi_day.x
            psi_exp_i_jse[:, j] = psi_day_jse.x

        Z[:, :, t] = R_exper - np.matmul(Bstar, psi_exp_i)   # Residuals for regular PCA
        Z_jse[:, :, t] = R_exper - np.matmul(Bstar_JSE, psi_exp_i_jse)  # Residuals for JSE

    return Z, Z_jse



# create simulationGPS object, which
# creates beta, factor, specific, total returns  x numExperiments

rng = np.random.default_rng()  # makes a random number generator, random seed

sim = sjse.SimulationJSE(rng, NormalFlag, MaxAssets, NumExperiments, NumPeriods, BetaMean, BetaStDev, Factor1StDev,
                         SpecificStDev, FactorFlag, Factor2StDev, Factor3StDev, Factor4StDev)

# get returns matrix (assets x periods x numExperiments) from sim object
Rtot = sim.GetReturnsMatrix() #if our factor flag is now 1, we produce 4 factor returns

# get the residuals matrix (assets x periods x numeExperiments) 
if FactorFlag == 1: 
    Z, Z_jse = Compute_Zmatrix(Rtot)
# get true betas -- one beta for all experiments
betaVector = sim.GetBetaVector()

# set up arrays to hold results of metrics per trial

VFR_raw = np.zeros(NumExperiments)  # var forecast ratio
VFR_Epca = np.zeros(NumExperiments)  # var forecast ratio
VFR_Ejse = np.zeros(NumExperiments)  # var forecast ratio

TrueVarR_Epca = np.zeros((NumExperiments))  # ratio of true min var to true var of est portfolio
TrueVarR_Ejse = np.zeros((NumExperiments))
TrueVarR_raw = np.zeros((NumExperiments))

trerrorraw = np.zeros((NumExperiments))  # pca tracking error
trerrorEpca = np.zeros(NumExperiments)  # pca tracking error
trerrorEjse = np.zeros((NumExperiments))  # gps tracking error

trueVar_Epca = np.zeros((NumExperiments))  # true variance of the PCA portfolio (min var with PCA cov estimate)
trueVar_Ejse = np.zeros((NumExperiments))  # true variance of the GPS portfolio
trueVar_raw = np.zeros((NumExperiments))
trueVar_TT = np.zeros((NumExperiments))

estVar_Epca = np.zeros((NumExperiments))  # estimated variance of the PCA portfolio (min var with PCA cov estimate)
estVar_Ejse = np.zeros((NumExperiments))  # est variance of the GPS portfolio
estVar_raw = np.zeros((NumExperiments))

###
# main loop iterating trials

for exper in range(NumExperiments):
    Y = Rtot[:, :, exper]  # matrix of returns for trial exper
    S = np.matmul(Y, Y.transpose()) / NumPeriods  # sample covariance matrix for trial exper
    b = betaVector / la.norm(betaVector)  # normalized beta
     
    
    # tracking error notation and formulas from MAPS paper, section 3


    ### we use these to compute the weights of the portfolio ###

    if FactorFlag == 0:
        h, h_GPS, sp2, lp2 = ComputePCA_GPS(S, NumPeriods, MaxAssets)  # defined above
        p_eta_true = np.dot(betaVector, betaVector) * (Factor1StDev) ** 2
        p_eta_obs = sp2 - lp2
        delta2_true = (SpecificStDev) ** 2
        delta2_obs = (NumPeriods / MaxAssets) * lp2

        delta2_raw = ((NumPeriods - 1) / (MaxAssets - 1)) * lp2 
        p_eta_raw = sp2 - delta2_raw

        w_Epca = ComputeMRPortfolio(h, p_eta = p_eta_obs, delta2 = delta2_obs)  # estimated evalues and PCA evector
        w_Ejse = ComputeMRPortfolio(h_GPS, p_eta = p_eta_obs, delta2 = delta2_obs)  # estimated evalues and GPS evector
        w_raw = ComputeMRPortfolio(h, p_eta = p_eta_raw, delta2 = delta2_raw)  # PCA evalue and evector

        w_TT = ComputeMRPortfolio(b, p_eta = p_eta_true, delta2 = delta2_true)  # true optimal portfolio
    
    elif FactorFlag == 1: 
        h, h_GPS, sp2 = ComputePCA_GPS(S, NumPeriods, MaxAssets)  # defined above
        true_mvar = np.array([Factor1StDev**2, Factor2StDev**2, Factor3StDev**2, Factor4StDev**2])
        z = Z[:, :, exper]
        z = np.sum(z**2, axis = 1) # square and sum across time
        d2n = (1/NumPeriods)*z # this is the calculation for equation (21) in better betas
        d2n_avg = np.sum(d2n)/len(d2n) # (22) bb
        sigma2 = sp2[0] - d2n_avg # (23) bb
        d2_mp = (np.trace(S) - np.sum(sp2))/(MaxAssets - 4*(1 - MaxAssets/NumPeriods)) # Marchenko-Pastur correction
        sigma2_mp = sp2[0] - d2_mp*(1 + MaxAssets/NumPeriods) # Marchenko-Pastur correction for market variance
        
        w_Epca = ComputeMRPortfolio(h, mvar = sigma2_mp, svar = d2_mp)  # weights 4 factor corrected market and specific variance
        w_Ejse = ComputeMRPortfolio(h_GPS, mvar = sigma2_mp, svar = d2_mp)  # weights 4 factor corrected beta vector and variances
        w_raw = ComputeMRPortfolio(h, mvar = sigma2, svar = d2n)  # weights 4 factor regular variance estimates as well as regular beta vector

        w_TT = ComputeMRPortfolio(p_eta_true, delta2_true, b)  # true optimal portfolio

        
    # see the JS# paper for the raw PCA estimator

    # four portfolios

    

    # tracking error, daily

    TrackErr2_Epca = p_eta_true * ((np.dot(w_Epca - w_TT, b)) ** 2) + delta2_true * ((la.norm(w_Epca - w_TT)) ** 2)

    TrackErr2_Ejse = p_eta_true * ((np.dot(w_Ejse - w_TT, b)) ** 2) + delta2_true * ((la.norm(w_Ejse - w_TT)) ** 2)

    TrackErr2_raw = p_eta_true * ((np.dot(w_raw - w_TT, b)) ** 2) + delta2_true * ((la.norm(w_raw - w_TT)) ** 2)

    # annualized, and reported as percent

    trerrorEpca[exper] = np.sqrt(252 * TrackErr2_Epca) * 100
    trerrorEjse[exper] = np.sqrt(252 * TrackErr2_Ejse) * 100
    trerrorraw[exper] = np.sqrt(252 * TrackErr2_raw) * 100

    # variance forecast ratios = estimated / true variances of the estimated portfolio
    # see GPS and MAPS for these variance formulas

    trueVar_Epca[exper] = p_eta_true * ((np.dot(b, w_Epca)) ** 2) + delta2_true * np.dot(w_Epca, w_Epca)
    trueVar_Ejse[exper] = p_eta_true * ((np.dot(b, w_Ejse)) ** 2) + delta2_true * np.dot(w_Ejse, w_Ejse)
    trueVar_raw[exper] = p_eta_true * ((np.dot(b, w_raw)) ** 2) + delta2_true * np.dot(w_raw, w_raw)
    trueVar_TT[exper] = p_eta_true * ((np.dot(b, w_TT)) ** 2) + delta2_true * np.dot(w_TT, w_TT)

    estVar_Epca[exper] = p_eta_obs * ((np.dot(h, w_Epca)) ** 2) + delta2_obs * np.dot(w_Epca, w_Epca)
    estVar_Ejse[exper] = p_eta_obs * ((np.dot(h_GPS, w_Ejse)) ** 2) + delta2_obs * np.dot(w_Ejse, w_Ejse)
    estVar_raw[exper] = p_eta_raw * ((np.dot(h, w_raw)) ** 2) + delta2_raw * np.dot(w_raw, w_raw)

    VFR_Epca[exper] = estVar_Epca[exper] / trueVar_Epca[exper]
    VFR_Ejse[exper] = estVar_Ejse[exper] / trueVar_Ejse[exper]
    VFR_raw[exper] = estVar_raw[exper] / trueVar_raw[exper]

    # true variance ratios

    TrueVarR_Epca[exper] = trueVar_TT[exper] / trueVar_Epca[exper]
    TrueVarR_Ejse[exper] = trueVar_TT[exper] / trueVar_Ejse[exper]
    TrueVarR_raw[exper] = trueVar_TT[exper] / trueVar_raw[exper]

# endfor exper


### output data files #######################################

np.savetxt(DayString + 'trerrorRaw.out', trerrorraw, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'trerrorEpca.out', trerrorEpca, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'trerrorEjse.out', trerrorEjse, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'VFR_raw.out', VFR_raw, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'VFR_Epca.out', VFR_Epca, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'VFR_Ejse.out', VFR_Ejse, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'TrueVarR_raw.out', TrueVarR_raw, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'TrueVarR_Epca.out', TrueVarR_Epca, fmt='%.3e', delimiter=',')
np.savetxt(DayString + 'TrueVarR_Ejse.out', TrueVarR_Ejse, fmt='%.3e', delimiter=',')

### output box plots  ######################################

column_names = ['raw', 'PCA', 'JSE']

#### tracking error

data_tr = np.array([trerrorraw, trerrorEpca, trerrorEjse]).transpose()

df_tr = pd.DataFrame(data_tr, columns=column_names)

FigTrE = plt.figure()
bp_tr = df_tr.boxplot()
plt.ylabel("annualized tracking error (%)")
plt.xlabel("estimator")
plt.title("Tracking Error " + str(NumExperiments) + " Experiments")
FigTrE.savefig(DayString + "trerrorboxplot_E" + str(NumExperiments) + "T" + str(NumPeriods) + "f" + str(
    NormalFlag) + ".pdf", format="pdf", bbox_inches="tight")

#### variance forecast ratio

data_vf = np.array([VFR_raw, VFR_Epca, VFR_Ejse]).transpose()

df_vf = pd.DataFrame(data_vf, columns=column_names)

FigVF = plt.figure()
bp_vf = df_vf.boxplot()
plt.ylabel("variance forecast ratio")
plt.xlabel("estimator")
FigVF.savefig(DayString + "varFR_boxplot_E" + str(NumExperiments) + "T" + str(NumPeriods) + "f" + str(
    NormalFlag) + ".pdf", format="pdf", bbox_inches="tight")

#### true variance ratios

data_var = np.array([TrueVarR_raw, TrueVarR_Epca, TrueVarR_Ejse]).transpose()

df_var = pd.DataFrame(data_var, columns=column_names)

FigVar = plt.figure()
bp_var = df_var.boxplot()
plt.ylabel("true variance ratio")
plt.xlabel("estimator")

FigVar.savefig(DayString + "trueVarRatio_boxplot_E" + str(NumExperiments) + "T" + str(NumPeriods) + "f" + str(
        NormalFlag) + ".pdf", format="pdf", bbox_inches="tight")

print("just testing lol")