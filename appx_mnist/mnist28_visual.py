import pandas as pd
import numpy as np
import random
import torch
from sdv.tabular import CTGAN, TVAE
from cDCGAN import cDCGAN
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

r = robjects.r
base = rpackages.importr('base')

# If not yet installed, please install the R packages in the comment block below
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=37)
# utils.install_packages('ranger')
# utils.install_packages('doParallel')
# utils.install_packages('data.table')
# utils.install_packages('arf')

arf = rpackages.importr('arf')
doPar = rpackages.importr('doParallel')


pandas2ri.activate()
doPar.registerDoParallel(20)

torch.backends.cudnn.deterministic = True

# Renew seeds for all used packages
def renewSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base.set_seed(seed,kind = "L'Ecuyer-CMRG")

# Plot example images and save to PNG
def visualize(X, svName=None):
    X = X.groupby('label').sample(10)
    X = X.iloc[:,:-1].values.reshape(100,28,28)
    fig, axes = plt.subplots(10, 10,figsize=(28,28))
    for i in range(100):
        ax = axes[i//10, i%10]
        ax.imshow(X[i],cmap='gray')
        ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    if svName is not None:
        plt.savefig("results/" + svName +".png", pad_inches = 0, bbox_inches='tight')
    plt.close()


renewSeed(2023)
X = pd.read_csv('mnist28.csv')
visualize(X, svName="mnist28")

# ARF - original version (not used for result)
renewSeed(2023)
X_ARF = X.copy()
X_ARF.iloc[:,:] = X_ARF.iloc[:,:].astype('str')
arf_rf = arf.adversarial_rf(X_ARF)
arf_dens = arf.forde(arf_rf,X_ARF)
Xfake_ARF = arf.forge(arf_dens,10000)
Xfake_ARF = (pandas2ri.PandasDataFrame(Xfake_ARF,index=Xfake_ARF.colnames).transpose()).astype('int')
Xfake_ARF.to_csv('results/mnist28_ARF.csv', index=False)
visualize(Xfake_ARF, svName="mnist28_ARF")

# import CSV for modified ARF data (run in R previously)
renewSeed(2023)
X_fake_ARF = pd.read_csv('results/mnist28_ARF.csv')
X_fake_ARF = X_fake_ARF.iloc[:,1:]
visualize(X_fake_ARF, svName="mnist28_ARF")

# TVAE
renewSeed(2023)
model_TVAE = TVAE()
model_TVAE.fit(X)
Xfake_TVAE = model_TVAE.sample(10000)
Xfake_TVAE.to_csv('results/mnist28_TVAE.csv', index=False)
visualize(Xfake_TVAE, svName="mnist28_TVAE")

# CTGAN
renewSeed(2023)
model_CTGAN = CTGAN()
model_CTGAN.fit(X)
Xfake_CTGAN = model_CTGAN.sample(10000)
Xfake_CTGAN.to_csv('results/mnist28_CTGAN.csv', index=False)
visualize(Xfake_CTGAN, svName="mnist28_CTGAN")

# cDCGAN
renewSeed(2022)
X_cDCGAN = X.iloc[:,:-1].copy()
X_cDCGAN = X_cDCGAN.apply(lambda x: (x - x.mean()),axis=1)
y_cDCGAN = X.iloc[:,-1].copy()

model_cDCGAN = cDCGAN(NUM_EPOCH=9, BATCH_SIZE=64, Adam_beta1=0)
model_cDCGAN.fit(X_cDCGAN,y_cDCGAN)
Xfake_cDCGAN = model_cDCGAN.sample(10000)
Xfake_cDCGAN.to_csv('results/mnist28_cDCGAN_.csv', index=False)
visualize(Xfake_cDCGAN,svName='mnist28_cDCGAN_')
