cd /apps/adc/act/quicklooks/dailyquicklooks/
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate plotting_env

# TODO: Move the txt files out of the main directory after executing act_plotting_main.py to clean it up a bit

#ls /data/archive/sgp/ | grep -vE '.a1|.a0|.00' > sgp.txt
#ls /data/archive/nsa/ | grep -vE '.a1|.a0|.00' > nsa.txt
#ls /data/archive/ena/ | grep -vE '.a1|.a0|.00' > ena.txt
#ls /data/archive/mos/ | grep -vE '.a1|.a0|.00' > mos.txt
#ls /data/archive/cor/ | grep -vE '.a1|.a0|.00' > cor.txt
#ls /data/archive/anx/ | grep -vE '.a1|.a0|.00' > anx.txt
#ls /data/archive/oli/ | grep -vE '.a1|.a0|.00' > oli.txt
ls /data/archive/guc/ | grep -vE '.00' > guc.txt
#ls /data/archive/hou/ | grep -vE '.00' > hou.txt

python act_plotting_main.py \
  -p -ts \
  --num-threads 20 \
  --num-days 120 \
  --base-out-dir /var/ftp/quicklooks/ \
  --max-file-size 200000000 \
  --use-txt-dir \
  -- index \
  >> logs/act.$(date +\%Y\%m\%d).log 2>&1

