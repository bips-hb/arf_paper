try:
    exec(open("runtime_main.py").read())
except:
    pass

exec(open("runtime_main.py").read())
# run separately to avoid memory issues

# for subsample benchmark: 
#sub_0 = run_CTGAN_cpu_sub(range_i = range(8))
#pd.concat(sub_0).to_csv("CTGAN_cpu-0.csv")
#sub_1 = run_CTGAN_cpu_sub(range_i = range(8,16) )
#pd.concat(sub_1).to_csv("CTGAN_cpu-1.csv")
#sub_2 = run_CTGAN_cpu_sub(range_i = range(16,24))
#pd.concat(sub_2).to_csv("CTGAN_cpu-2.csv")
#sub_3 = run_CTGAN_cpu_sub(range_i = range(24, 32) )
#pd.concat(sub_3).to_csv("CTGAN_cpu-3.csv")
#sub_4 = run_CTGAN_cpu_sub(range_i = range(32,40) )
#pd.concat(sub_4).to_csv("CTGAN_cpu-4.csv")

# for dimensionality benchmark: 
sub_0 = run_CTGAN_cpu_sub(range_i = range(7))
pd.concat(sub_0).to_csv("CTGAN_cpu-0.csv")
sub_1 = run_CTGAN_cpu_sub(range_i = range(7,14) )
pd.concat(sub_1).to_csv("CTGAN_cpu-1.csv")
sub_2 = run_CTGAN_cpu_sub(range_i = range(14,21))
pd.concat(sub_2).to_csv("CTGAN_cpu-2.csv")
sub_3 = run_CTGAN_cpu_sub(range_i = range(21, 28) )
pd.concat(sub_3).to_csv("CTGAN_cpu-3.csv")
sub_4 = run_CTGAN_cpu_sub(range_i = range(28,35) )
pd.concat(sub_4).to_csv("CTGAN_cpu-4.csv")