# cd /opt/ml/wxcode

python -u src/main_all.py \
    --seed 2022 \
    --savedmodel_path save/v1
    
python -u src/main_all.py \
    --seed 2021 \
    --savedmodel_path save/v2 
    
python -u src/main_all.py \
    --seed 2020 \
    --savedmodel_path save/v3
