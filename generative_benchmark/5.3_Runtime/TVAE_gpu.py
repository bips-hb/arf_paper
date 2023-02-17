try:
    exec(open("runtime_main.py").read())
except:
    pass

exec(open("runtime_main.py").read())

pd.concat(run_sub(synthesizer_name= "TVAE_gpu", R_seed = False)).to_csv("TVAE_gpu.csv")