
from gefs import RandomForest
import numpy as np

def gefs_fun(trn_x, trn_y, tst_dat, ncat, n_estimators):
  rf = RandomForest(n_estimators=n_estimators, ncat=ncat)  # Train a Random Forest
  rf.fit(trn_x, trn_y)
  gef = rf.topc()  # Convert to a GeF
  return gef.log_likelihood(tst_dat)
