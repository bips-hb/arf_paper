import os
import pandas as pd
print(os.listdir())
import sys
print(sys.path)
sys.path.append("")
print(sys.path)
os.chdir("./ITGAN_adjusted2")
import train_itgan as ITGAN ## unzip ITGAN_adjusted2.zip to access this module
from util.data import load_dataset,get_metadata
os.chdir("..")
try:
    exec(open("benchmark_individual.py").read())
except:
    pass

exec(open("benchmark_individual.py").read())


# prepare ITGAN model

def itgan_fun(real_data):
    # find categorical columns
    cat_cols = real_data.select_dtypes('object').columns.to_list()
    ord_cols = []
    arg = ITGAN.getArgs(data = real_data, cat_col = cat_cols, ord_col = ord_cols, GPU_NUM=0)
    return ITGAN.AEGANSynthesizer(**arg)


# define synthetic data generation
def synth_data(data_train, synthesizer):
    """
    Arguments: 
    @data_train: data to learn synthesizer from
    @synthesizer: model for generating synthetic data
    Return: synthesized data of size data_train
    """
    if synthesizer == itgan_fun:
        syn = itgan_fun(real_data = data_train)
        syn.fit()
        return syn.sample(data_train.shape[0])
    else: 
        print("please use ITGAN synthesizer")



#####################
# ! for pretesting whether code runs in <24h, evaluate code on one replicate:
# ! evaluate datasets one by one -> comment out other datasets during pretests
# ! comment out the following line for single replicate 
#rep = range(1)

print(f"ITGAN sucessfully initialized, number of reps is {rep}")

# adult
adult_res = run_benchmark(training_data= adult_train, test_data = adult_test, classifiers= adult_classifiers, 
metrics= adult_metrics, data_synthesizer= {"ITGAN": itgan_fun})
pd.concat(adult_res).to_csv("ITGAN_adult.csv")

# census ## does not finish in 24h
#census_res = run_benchmark(training_data = census_train, test_data = census_test, classifiers = census_classifiers,
#metrics = census_metrics, data_synthesizer = {"ITGAN": itgan_fun})
#pd.concat(census_res).to_csv("ITGAN_census.csv")

# credit ## does not finish in 24h
#credit_res = run_benchmark(training_data= credit_train, test_data= credit_test, classifiers= credit_classifiers,
#metrics= credit_metrics, data_synthesizer= {"ITGAN": itgan_fun})
#pd.concat(credit_res).to_csv("ITGAN_credit.csv")

# covtype ## does not finish in 24h
#covtype_res = run_benchmark(training_data = covtype_train, test_data= covtype_test, classifiers= covtype_classifiers,
#metrics= covtype_metrics, data_synthesizer=  {"ITGAN": itgan_fun})
#pd.concat(covtype_res).to_csv("ITGAN_covtype.csv")

# intrusion ## does not finish in 24h
#intrusion_res = run_benchmark(training_data= intrusion_train, test_data= intrusion_test, classifiers= intrusion_classifiers,
#metrics= intrusion_metrics, data_synthesizer= {"ITGAN": itgan_fun})
#pd.concat(intrusion_res).to_csv("ITGAN_intrusion.csv")