try:
    exec(open("runtime_main.py").read())
except:
    pass

exec(open("runtime_main.py").read())

doPar = rpackages.importr('doParallel')
doPar.registerDoParallel(10)

pd.concat(run_sub(synthesizer_name= "gen_rf", R_seed = True)).to_csv("grf_cpu.csv")