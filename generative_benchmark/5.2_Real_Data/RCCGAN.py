import os
print(os.getcwd())
print(os.listdir())
import sys
print(sys.path)
sys.path.append("")
print(sys.path)

try:
    exec(open("benchmark_individual.py").read())
except:
    pass

exec(open("benchmark_individual.py").read())

os.chdir("./RccGAN/RccGAN/ctgan")

import numpy as np
import pandas as pd
from ctgan import CTGANSynthesizer ## unzip RCCGAN.zip in  environment_files folder to access this module
from data import read_csv

os.chdir("../../..")
print(os.getcwd())



# define synthetic data generation
def synth_data(data_train, synthesizer= {'RCCGAN': 'RCCGAN'}):
    print('this function uses RCC-GAN to synthesize data')
    """
    Arguments: 
    @data_train: data to learn synthesizer from
    @synthesizer: model for generating synthetic data
    Return: synthesized data of size data_train
    """
    cat_col = data_train.select_dtypes('object').columns.to_list()
    cat_col_str = ','.join(cat_col)
    data, discrete_columns =(read_csv(data_frame=data_train, 
    list_of_all_variables = data_train.columns.tolist(),
    discrete= cat_col_str))
    generator_dim = [int(x) for x in '256,256'.split(',')]
  #  print(discrete_columns)
    discriminator_dim = [int(x) for x in '256,256'.split(',')]
    model = CTGANSynthesizer(
        embedding_dim=128, generator_dim=generator_dim,
        discriminator_dim=discriminator_dim, generator_lr=2e-4,
        generator_decay=1e-6, discriminator_lr=2e-4,
        discriminator_decay=0, batch_size=500,
        epochs=300)
    model.fit(data, discrete_columns=discrete_columns)
    
    return model.sample(len(data_train), None, None)



#####################
# ! for pretesting whether code runs in <24h, evaluate code on one replicate:
# ! evaluate datasets one by one -> comment out other datasets during pretests
# ! comment out the following line for single replicate 
#rep = range(1)

print(f"RCCGAN sucessfully initialized, number of reps is {rep}")

# adult
adult_res = run_benchmark(training_data= adult_train, test_data = adult_test, classifiers= adult_classifiers, 
metrics= adult_metrics, data_synthesizer= {"RCCGAN": 'RCCGAN'})
pd.concat(adult_res).to_csv("RCCGAN_adult.csv")

# census
census_res = run_benchmark(training_data = census_train, test_data = census_test, classifiers = census_classifiers,
metrics = census_metrics, data_synthesizer =  {"RCCGAN": 'RCCGAN'})
pd.concat(census_res).to_csv("RCCGAN_census.csv")

# credit
credit_res = run_benchmark(training_data= credit_train, test_data= credit_test, classifiers= credit_classifiers,
metrics= credit_metrics, data_synthesizer=  {"RCCGAN": 'RCCGAN'})
pd.concat(credit_res).to_csv("RCCGAN_credit.csv")


# covtype ## kills itself - memory issues ## does not finish in 24h
#covtype_res = run_benchmark(training_data = covtype_train, test_data= covtype_test, classifiers= covtype_classifiers,
#metrics= covtype_metrics, data_synthesizer=  {"RCCGAN": 'RCCGAN'})
#pd.concat(covtype_res).to_csv("RCCGAN_covtype.csv")

# intrusion ## does not finish in 24h
#intrusion_res = run_benchmark(training_data= intrusion_train, test_data= intrusion_test, classifiers= intrusion_classifiers,
#metrics= intrusion_metrics, data_synthesizer= {"RCCGAN": 'RCCGAN'})
#pd.concat(intrusion_res).to_csv("RCCGAN_intrusion.csv")


