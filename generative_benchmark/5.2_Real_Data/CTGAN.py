try:
    exec(open("benchmark_individual.py").read())
except:
    pass

exec(open("benchmark_individual.py").read())
from sdv.tabular import CTGAN


# adult
adult_res = run_benchmark(training_data=adult_train, test_data=adult_test, classifiers=adult_classifiers,
metrics= adult_metrics, data_synthesizer=  {"CTGAN": CTGAN()})
pd.concat(adult_res).to_csv("CTGAN_adult.csv")

# census
census_res = run_benchmark(training_data = census_train, test_data = census_test, classifiers = census_classifiers,
metrics = census_metrics, data_synthesizer = {"CTGAN": CTGAN()})
pd.concat(census_res).to_csv("CTGAN_census.csv")

# covtype
covtype_res = run_benchmark(training_data = covtype_train, test_data= covtype_test, classifiers= covtype_classifiers,
metrics= covtype_metrics, data_synthesizer= {"CTGAN": CTGAN()})
pd.concat(covtype_res).to_csv("CTGAN_covtype.csv")

# credit
credit_res = run_benchmark(training_data= credit_train, test_data= credit_test, classifiers= credit_classifiers,
metrics= credit_metrics, data_synthesizer= {"CTGAN": CTGAN()})
pd.concat(credit_res).to_csv("CTGAN_credit.csv")

# intrusion
intrusion_res = run_benchmark(training_data= intrusion_train, test_data= intrusion_test, classifiers= intrusion_classifiers,
metrics= intrusion_metrics, data_synthesizer= {"CTGAN": CTGAN()})
pd.concat(intrusion_res).to_csv("CTGAN_intrusion.csv")

# mnist12
mnist12_res = run_benchmark(training_data= mnist12_train,test_data= mnist12_test, classifiers= mnist12_classifiers,
metrics= mnist12_metrics, data_synthesizer= {"CTGAN": CTGAN()})
pd.concat(mnist12_res).to_csv("CTGAN_mnist12.csv")

# mnist28
mnist28_res = run_benchmark(training_data= mnist28_train, test_data= mnist28_test, classifiers= mnist28_classifiers,
metrics= mnist28_metrics, data_synthesizer= {"CTGAN": CTGAN()})
pd.concat(mnist28_res).to_csv("CTGAN_mnist28.csv")
