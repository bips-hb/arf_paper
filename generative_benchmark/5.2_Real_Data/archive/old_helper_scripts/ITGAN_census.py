import os
import pandas as pd
print(os.listdir())
import sys
print(sys.path)
sys.path.append("")
print(sys.path)
os.chdir("./ITGAN_adjusted2")
import train_itgan as ITGAN
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
        print("ITGAN has synthesized")
        return syn.sample(data_train.shape[0])
    else: 
        print("please use ITGAN synthesizer")


# example
#url = 'https://raw.githubusercontent.com/sdv-dev/CTGAN/master/examples/csv/adult.csv'
#data = pd.read_csv(url)
#ex = itgan_fun(data)

#bb = synth_data(data_train=dd, synthesizer=itgan_fun)

# more examples on subsets

#census_train = list(census_train)
#census_train[0] = census_train[0].iloc[1:5000,]
#census_train = tuple(census_train)
#census_test = list(census_test)
#census_test[0] = census_test[0].iloc[1:5000,]
#census_test = tuple(census_test)
#census_train[0].name = 'census'
#census_test[0].name = 'census'

#dd= census_train[0].copy()
#oo = synth_data(data_train=dd, synthesizer=itgan_fun)
#for col in dd.select_dtypes('object').columns: print(dd[col].unique())

#####################
# !!!! only one rep, choose first data instance of adult, census, etc.

rep = range(1)

print(f"ITGAN sucessfully initialized, number of reps is {rep}")

# adult
#adult_res = run_benchmark(training_data= adult_train, test_data = adult_test, classifiers= adult_classifiers, 
#metrics= adult_metrics, data_synthesizer= {"ITGAN": itgan_fun})
#pd.concat(adult_res).to_csv("ITGAN_adult.csv")

# census
census_res = run_benchmark(training_data = census_train, test_data = census_test, classifiers = census_classifiers,
metrics = census_metrics, data_synthesizer = {"ITGAN": itgan_fun})
pd.concat(census_res).to_csv("ITGAN_census.csv")

# credit
#credit_res = run_benchmark(training_data= credit_train, test_data= credit_test, classifiers= credit_classifiers,
#metrics= credit_metrics, data_synthesizer= {"ITGAN": itgan_fun})
#pd.concat(credit_res).to_csv("ITGAN_credit.csv")

