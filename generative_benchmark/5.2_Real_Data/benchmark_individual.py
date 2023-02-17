import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sdgym.datasets import load_dataset
from sdgym.datasets import load_tables
from sklearn.metrics import f1_score, accuracy_score, r2_score
from sdv.metrics.tabular import BinaryDecisionTreeClassifier,BinaryAdaBoostClassifier,BinaryLogisticRegression,BinaryMLPClassifier, MulticlassDecisionTreeClassifier, MulticlassMLPClassifier
import torch

def oracle(): 
    pass 

def f1_none(*args):
  return f1_score(average=None, *args)

def f1_macro(*args):
  return f1_score(average = 'macro', *args)

def f1_micro(*args):
  return f1_score(average = 'micro', *args)

def synth_data(data_train, synthesizer):
    """
    Arguments:
    @data_train: data to learn synthesizer from
    @synthesizer: model for generating synthetic data
    Return: synthesized data of size data_train
    """     
    if synthesizer == oracle():
       return data_train.copy()
  #  elif synthesizer == gen_rf:
   #     return gen_rf(real_data = data_train)
  #  elif synthesizer == gen_rf_oob:
   #     return gen_rf_oob(real_data = data_train)
    else: 
        synthesizer.fit(data = data_train)
        return synthesizer.sample(data_train.shape[0])  

def scores(data_train, data_test, list_of_classifiers, metric, synthesizer):
    """
    Args: 
    - list_of_classifiers: list of classifiers for the prediction task, subset of [BinaryDecisionTreeClassifier, BinaryAdaBoostClassifier,BinaryLogisticRegression, BinaryMLPClassifier, LinearRegression, MLPRegressor]
    - metric: metric to use for score subset of [f1_none, accuracy_score, r2_score]
    - synthesizer: dict of synthesizer to generate synthetic data
    Returns: scores
    """ 
    syn_dat_res = pd.DataFrame()
    wall_time_start, process_time_start = time.perf_counter(), time.process_time()
    syn_data = synth_data(data_train = data_train, synthesizer = synthesizer[list(synthesizer.keys())[0]])
    wall_time, process_time = time.perf_counter()-wall_time_start, time.process_time()-process_time_start
    res = pd.DataFrame()
    for item in list_of_classifiers:
        scores = item.compute(real_data = data_test, synthetic_data = syn_data, target = "label", scorer = metric)
        new_metric = pd.DataFrame()
        for i in range(len(metric)): new_metric = pd.concat([new_metric, pd.DataFrame({metric[i].__name__ : [scores[i]] })], axis=1)
        res = res.append(pd.concat([pd.DataFrame({'dataset': data_train.name, 'model': list(synthesizer.keys())[0],'classifier': [item.__name__] , 
        'wall_time':wall_time, 'process_time':process_time}), new_metric], axis = 1))
    syn_dat_res = syn_dat_res.append(res)
    return syn_dat_res

rep = range(5)

def run_benchmark(training_data, test_data, classifiers, metrics, data_synthesizer, seed = 2022):
    np.random.seed(seed)
    torch.manual_seed(seed)
    comp = (scores(data_train = training_data[i], data_test = test_data[i], list_of_classifiers = classifiers, 
    metric = metrics, synthesizer = data_synthesizer) for i in rep)
    return list(comp)
################################
# get indices of train/test data sets for each data set
################################

np.random.seed(2022)

# adult 
adult = load_tables(load_dataset('adult'))['adult']
adult_train, adult_test = zip(*[train_test_split(adult, test_size=10/(23+10), stratify=adult['label']) for i in rep])
for i in rep:
  adult_train[i].name = 'adult'  # keep information on which dataset is used
adult_classifiers = [BinaryDecisionTreeClassifier,BinaryAdaBoostClassifier,BinaryLogisticRegression,BinaryMLPClassifier]
adult_metrics = [f1_none, accuracy_score]

# census 
census = load_tables(load_dataset('census'))['census']
census = census.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) 
census_train, census_test = zip(*[train_test_split(census, test_size=100/(100+200), stratify=census['label']) for i in rep])
for i in rep:
  census_train[i].name = 'census'
census_classifiers = [BinaryDecisionTreeClassifier,BinaryAdaBoostClassifier,BinaryMLPClassifier]
census_metrics = [f1_none, accuracy_score]

# covtype
covtype = load_tables(load_dataset('covtype'))['covtype']
covtype_train, covtype_test = zip(*[train_test_split(covtype, test_size=100/(481+100), stratify=covtype['label'])for i in rep])
for i in rep:
  covtype_train[i].name = 'covtype'
covtype_classifiers = [MulticlassDecisionTreeClassifier,MulticlassMLPClassifier]
covtype_metrics = [accuracy_score, f1_micro, f1_macro]

# credit
credit = load_tables(load_dataset('credit'))['credit']
credit_train, credit_test = zip(*[train_test_split(credit, test_size=20/(264+20), stratify=credit['label'])for i in rep])
for i in rep:
  credit_train[i].name = 'credit'
credit_classifiers = [BinaryDecisionTreeClassifier,BinaryAdaBoostClassifier,BinaryMLPClassifier]
credit_metrics = [f1_none, accuracy_score]

# intrusion
intrusion = load_tables(load_dataset('intrusion'))['intrusion']
intrusion.drop('is_host_login', axis = 1)
intrusion_train, intrusion_test = zip(*[train_test_split(intrusion, test_size=100/(394+100), stratify=intrusion['label'])for i in rep])
for i in rep:
  intrusion_train[i].name = 'intrusion'
intrusion_classifiers = [MulticlassDecisionTreeClassifier,MulticlassMLPClassifier]
intrusion_metrics = [accuracy_score, f1_micro, f1_macro] 

# mnist12
mnist12 = load_tables(load_dataset('mnist12'))['mnist12']
mnist12_train, mnist12_test = zip(*[train_test_split(mnist12, test_size=10/(60+10), stratify=mnist12['label'])for i in rep])
for i in rep:
  mnist12_train[i].name = 'mnist12'
mnist12_classifiers = [MulticlassDecisionTreeClassifier,MulticlassMLPClassifier]
mnist12_metrics = [accuracy_score, f1_micro, f1_macro]

# mnist28
mnist28 = load_tables(load_dataset('mnist28'))['mnist28']
mnist28_train, mnist28_test = zip(*[train_test_split(mnist28, test_size=10/(60+10), stratify=mnist28['label'])for i in rep])
for i in rep:
  mnist28_train[i].name = 'mnist28'
mnist28_classifiers = [MulticlassDecisionTreeClassifier,MulticlassMLPClassifier]
mnist28_metrics = [accuracy_score, f1_micro, f1_macro]
