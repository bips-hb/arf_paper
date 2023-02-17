# this is a special runtime_main.py script for CTABGAN
# basically the same code as runtime_main.py, but with different imports (because of environment issues)
# and some special adjustments for CTABGAN (e.g. write out every subset results as .csv)

import time
import pandas as pd
import numpy as np
from sdgym.datasets import load_dataset
from sdgym.datasets import load_tables
from sklearn.utils import resample

import torch
from sdv.tabular import TVAE
from sdv.tabular import CTGAN

# load adult data
adult = load_tables(load_dataset('adult'))['adult']
# basically same syn_time function as in runtime_main.py, but with some special adjustments for CTABGAN,
#  e.g. writing out .csv files for each subset
def syn_time(data, synthesizer, **kwargs):
    """
    Args: 
    - data: real data to train data synthesizer on
    - synthesizer: dict of synthesizer to be trained
    Returns: walltime and process time to train synthesizer 
    """ 
    if list(synthesizer.keys())[0] in ['CTABGAN_gpu' , 'CTABGAN_cpu']:
        print(f"THIS IS {list(synthesizer.keys())[0]}" )
        # find categorical columns
        cat_col = data.select_dtypes('object').columns.to_list()
        # find integer columns
        int_col = data.select_dtypes('int').columns.to_list()
        # define CTABGAN model
        mod =  CTABGAN(df = data,
        test_ratio = 0.2,
        categorical_columns = cat_col,
        integer_columns = int_col,
        problem_type = { None: None})
        # start time measurement
        wall_time_start, process_time_start = time.perf_counter(), time.process_time()
        # fit model and measure time
        mod.fit()
        wall_time_train, process_time_train = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
        # sample from model
        wall_time_start, process_time_start = time.perf_counter(), time.process_time()
        wall_time_sample, process_time_sample = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
        try: ## for some dimensionalities sampling raises an error -- set sampling time to None for these cases
          mod.generate_samples()
          wall_time_sample, process_time_sample = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
        except:
          wall_time_sample, process_time_sample = None, None

    elif list(synthesizer.keys())[0] == 'gen_rf':
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
    if list(synthesizer.keys())[0] in ['CTABGAN_gpu' , 'CTABGAN_cpu']:
      res.to_csv(f"res_{list(synthesizer.keys())[0]}_sub-{kwargs['run']}.csv")
    return res


def run_CTABGAN_sub(range_i, synthesizer_name):
    np.random.seed(2022)
    torch.manual_seed(2022)
    my_syn = []
    i = 0
    while i < len(subs):
      if synthesizer_name == "CTABGAN_gpu":
        my_syn.append({"CTABGAN_gpu": None})
      elif synthesizer_name == "CTABGAN_cpu":
        my_syn.append({"CTABGAN_cpu": None})
      else: 
        print("please specify whether CTABGAN is run on GPU or CPU")
      i = i+1
    res = (syn_time(data = data_sub[i], synthesizer =  my_syn[i], run = i) for i in range_i)
    return list(res)

###
# SELECT WHICH BENCHMARK TO RUN
# COMMENT OUT OTHER BENCHMARK
# same subsets as in runtime_main.py is guaranteed through setting np.random.seed(2022) right before subsample draw
###

#########
# for sample size benchmark, use
#########
# subs_log = np.exp(np.linspace(np.log(1000), np.log(32561), 8))
# subs = [round(i) for i in subs_log]*5
# ## sample subsets from data
# np.random.seed(2022)
# data_sub= [resample(adult, n_samples=subs[i], replace=False, stratify=adult['label']) for i in range(len(subs))] 


##########
# for dimensionality benchmark, use
##########
np.random.seed(2022)
# adult dtypes: 6 times 'int64', 9 times 'object' (including 'label')
# adult continuous features:
adult_cont = adult.select_dtypes(include='int64')
# adult categorical features: without 'label'
adult_cat = adult.select_dtypes(include='object').drop('label', axis=1)
# select one continous, one categorical and the label for subsets
rep = 5
subs = [2,4,6,8,10,12]*rep
data_sub = [pd.concat([adult_cont.sample(int(i/2), axis=1),adult_cat.sample(int(i/2), axis=1), adult['label']], axis=1) for i in subs]

i = 0
while i < rep:
 data_sub.append(adult)  # add full data set
 subs.append(14)
 i = i+1

