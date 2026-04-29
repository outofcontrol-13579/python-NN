import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats  # for p value using t distribution
from scipy.stats import t  # for t distribution in confidence intervalls
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm


def linreg(output, predictor):
  # assume Y = beta@X + err with 𝐸[err]=0 and variance  𝐸[err2]=𝜎2
  n = predictor.shape[0]
  p = predictor.shape[1]
  y = output
  # add intercept column
  X = np.c_[np.ones(n), predictor]
  # ols estimates for regression coefficients
  betahat = np.linalg.pinv(X) @ y
  # residuals and what the ols estimates minimize: rss
  # rss is used in Ftest to compare two fits
  residuals = y - X @ betahat.T
  rss = np.dot(residuals, residuals)
  l2_loss = 0.5 * rss / n
  # estimate of std of err: rse
  varerrhat = rss / (n - p - 1)
  rse = np.sqrt(varerrhat)
  # standard errors for ols estimates
  vcovbeta = varerrhat * np.linalg.inv(X.T @ X)
  stderrs = np.sqrt(np.diagonal(vcovbeta))
  # confidence intervals for ols estimates
  # should contain the 97.5% quantile of a t-distribution with n−p-1 degrees of freedom (1.96 for snd)
  df = n - p - 1
  # to be able to vectorize, see pstat.py and https://www.stat.uchicago.edu/~yibi/teaching/stat224/L04.pdf
  cutoff = t.ppf(0.975, df, loc=0, scale=1)
  confintervs = np.array([betahat - np.multiply(stderrs, cutoff), betahat + np.multiply(stderrs, cutoff)])
  # how many of their std the ols estimates are away from zero: tstats
  tstats = np.divide(betahat, stderrs)
  pvals = stats.t.sf(np.abs(tstats), n - 2) * 2
  # total variance in the response data before regression
  tss = np.dot(y - y.mean(), y - y.mean())
  # fraction of variance explained: Rsquared
  Rsquared = 1 - (rss / tss)
  # is at least one ols estimate non-zero: Fstatistic (1 if no, >>1 if yes)
  # works well when p << n
  F = ((tss - rss) / p) / varerrhat
  # confirm calculation using statsmodel
  # model = sm.OLS(y, X); results = model.fit(); print(results.summary()) #print(summarize(results))
  df = pd.DataFrame(data={
      'coef': betahat, 'std err': stderrs, 't': tstats, 'P>|t|': pvals, '[0.025': confintervs[0], '0.975]': confintervs[1]})
  return df, betahat, stderrs, tstats, pvals, confintervs, rse, Rsquared, F, rss, residuals, l2_loss
