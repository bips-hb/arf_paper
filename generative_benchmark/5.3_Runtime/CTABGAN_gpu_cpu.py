try:
    exec(open("CTABGAN_runtime_main.py").read())
except:
    pass

exec(open("CTABGAN_runtime_main.py").read())
 
import os
print(os.getcwd())
os.chdir("..")
os.chdir("./CTABGAN")
print(os.getcwd())
print(os.listdir())
import sys
print(sys.path)
sys.path.append("")
print(sys.path)
from model.ctabgan import CTABGAN
os.chdir("..")


# IMPORTANT NOTES
# (I)
# CTABGAN developers hard coded the devide without the option to pass an argument,
# to use GPU: manually set in CTABGAN module: model/synthesizer/ctabgan_synthesizer in CTABGANSynthesizer class l. 360: self.device = torch.device("cuda:0")
# to use CPU: manually set in  CTABGAN module: model/synthesizer/ctabgan_synthesizer in CTABGANSynthesizer class l. 360: self.device = "cpu"
# (II)
# for dimensionality = 3, integer column creation leads to an error in the inverse transformation -> results in NA for sampling time
# (III)
# as with CTGAN, we run into memory issues if we run all subsample/dimensionalitiy subsets in a list comprehension or for loop
# therefore, we use a specialised function, run_CTABGAN_sub(), with smaller range of subsets i and write out .csv files for each subset 


# GPU benchmark:
# for subsample benchmark:
# os.chdir("./runtime/results_samplesize")
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu', range_i =  range(8))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu', range_i = range(8,16))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu', range_i = range(16,24))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu', range_i = range(24,32))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu', range_i = range(32,40))

# for dimensionality benchmark:
os.chdir("./runtime/results_dimensionality")
run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu',range_i = range(7))
run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu',range_i = range(7,14))
run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu',range_i = range(14,21))
run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu',range_i = range(21, 28))
run_CTABGAN_sub(synthesizer_name= 'CTABGAN_gpu',range_i = range(28,35))


# CPU benchmark:
# for subsample benchmark:
# os.chdir("./runtime/results_samplesize")
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu', range_i =  range(8))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu', range_i = range(8,16))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu', range_i = range(16,24))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu', range_i = range(24,32))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu', range_i = range(32,40))

# for dimensionality benchmark:
# os.chdir("./runtime/results_dimensionality")
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu',range_i = range(7))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu',range_i = range(7,14))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu',range_i = range(14,21))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu',range_i = range(21, 28))
# run_CTABGAN_sub(synthesizer_name= 'CTABGAN_cpu',range_i = range(28,35))

