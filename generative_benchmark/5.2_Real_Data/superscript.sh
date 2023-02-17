# use this bash script for job submission
# adjust script by commenting out thing you do/don't want to run

#python grf_cpu.py
#python TVAE_gpu.py
#python TVAE_cpu.py
#python CTGAN_gpu.py
#python CTGAN_cpu.py
source /home/blesch/miniconda3/etc/profile.d/conda.sh
conda info --env
#conda activate ctabgan
#conda info --env
#python CTABGAN.py 
#conda activate gan
#conda info --env
#python RCCGAN.py
conda activate itgan
conda info --env
python ITGAN.py
#python ITGAN_adult.py
#timeout 24h python ITGAN_census.py
#timeout 24h python ITGAN_credit.py