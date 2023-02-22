import time
import pandas as pd
import numpy as np
from sdgym.datasets import load_dataset
from sdgym.datasets import load_tables
from sklearn.utils import resample
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r = robjects.r
genrf = rpackages.importr('genrf') ## old name for arf package ## delta = 0.5
gen_rf = ()
# assign registerDoParallel in grf_cpu.py file, therefore commented out here
#doPar = rpackages.importr('doParallel')
#doPar.registerDoParallel(10)
import torch
from sdv.tabular import TVAE
from sdv.tabular import CTGAN

# load adult data
adult = load_tables(load_dataset('adult'))['adult']

def syn_time(data, synthesizer):
    """
    Args: 
    - data: real data to train data synthesizer on
    - synthesizer: dict of synthesizer to be trained
    Returns: walltime and process time to train synthesizer 
    """ 
    if list(synthesizer.keys())[0] == 'gen_rf':
        wall_time_start, process_time_start = time.perf_counter(), time.process_time()
        mod = genrf.genrf['new'](data)
        wall_time_train, process_time_train = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
        wall_time_start, process_time_start = time.perf_counter(), time.process_time()
        mod['sample'](n = data.shape[0])
        wall_time_sample, process_time_sample = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
    else: 
        syn = synthesizer[list(synthesizer.keys())[0]]
        wall_time_start, process_time_start = time.perf_counter(), time.process_time()
        syn.fit(data = data)
        wall_time_train, process_time_train = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
        wall_time_start, process_time_start = time.perf_counter(), time.process_time()
        syn.sample(data.shape[0])  
        wall_time_sample, process_time_sample = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
    res = pd.DataFrame({'n': data.shape[0], 'd': data.shape[1], 'model': list(synthesizer.keys())[0],
    'wall_time_train':wall_time_train, 'process_time_train':process_time_train,
    'wall_time_sample':wall_time_sample, 'process_time_sample':process_time_sample}, index=[0])
    return res


def run_sub(synthesizer_name, R_seed = False):
    np.random.seed(2022)
    torch.manual_seed(2022)
    if R_seed:
        base = rpackages.importr('base')
        base.set_seed(2022,kind = "L'Ecuyer-CMRG")
        print("R seed set")
    my_syn = []
    i = 0
    while i < len(subs):
      if synthesizer_name == "TVAE_gpu":
        my_syn.append({"TVAE": TVAE(cuda=True)})
      elif synthesizer_name == "TVAE_cpu":
        my_syn.append({"TVAE": TVAE(cuda=False)})
      elif synthesizer_name == "CTGAN_gpu":
        my_syn.append({"CTGAN": CTGAN(cuda=True)})
      elif synthesizer_name == "CTGAN_cpu":
        my_syn.append({"CTGAN": CTGAN(cuda=False)})
      elif synthesizer_name == "gen_rf":
        my_syn.append({"gen_rf": gen_rf})
      else: 
        print("please specify synthesizer name")
      i=i+1
    res = (syn_time(data = data_sub[i], synthesizer =  my_syn[i]) for i in range(len(subs)))
    return list(res)

# for CTGAN cpu write own function to execute separately -- circumvent memory issues in list comprehension

def run_CTGAN_cpu_sub(range_i):
    np.random.seed(2022)
    torch.manual_seed(2022)
    my_syn = []
    i = 0
    while i < len(subs):
      my_syn.append({"CTGAN": CTGAN(cuda=False)})
      i = i+1
    res = (syn_time(data = data_sub[i], synthesizer =  my_syn[i]) for i in range_i)
    return list(res)


###
# SELECT WHICH BENCHMARK TO RUN
# COMMENT OUT OTHER BENCHMARK
###

#########
# for sample size benchmark, use
#########
#subs_log = np.exp(np.linspace(np.log(1000), np.log(32561), 8))
#subs = [round(i) for i in subs_log]*5
## sample subsets from data
#np.random.seed(2022)
#data_sub= [resample(adult, n_samples=subs[i], replace=False, stratify=adult['label']) for i in range(len(subs))] 


##########
# for dimensionality benchmark, use
##########
np.random.seed(2022)
## adult dtypes: 6 times 'int64', 9 times 'object' (including 'label')
## adult continuous features:
adult_cont = adult.select_dtypes(include='int64')
## adult categorical features: without 'label'
adult_cat = adult.select_dtypes(include='object').drop('label', axis=1)
## select one continous, one categorical and the label for subsets
rep = 5
subs = [2,4,6,8,10,12]*rep
data_sub = [pd.concat([adult_cont.sample(int(i/2), axis=1),adult_cat.sample(int(i/2), axis=1), adult['label']], axis=1) for i in subs]

i = 0
while i < rep:
  data_sub.append(adult)  # add full data set
  subs.append(14)
  i = i+1


#example =pd.concat(run_sub(synthesizer_name= "gen_rf", R_seed = True))