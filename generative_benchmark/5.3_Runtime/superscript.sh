source /home/blesch/miniconda3/etc/profile.d/conda.sh
conda info --env
conda activate renv
python grf_cpu.py
python TVAE_gpu.py
python TVAE_cpu.py
python CTGAN_gpu.py
python CTGAN_cpu.py
conda activate ctabgan
python CTABGAN_gpu_cpu.py ## switch from GPU to CPU usage in CTABGAN_gpu_cpu.py and run again: python CTABGAN_gpu_cpu.py 