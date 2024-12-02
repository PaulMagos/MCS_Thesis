<<<<<<< HEAD
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_SmallSeq12 -d -m python TrainingModel.py --size 8000 --batch_size 32 -lr 1e-5 --model_name AirQuality_SmallSeq4 -hs 16 --wandb -m 5 --epochs 500 --patience 50 --seq_length 4 --teacher_forcing`
exec `CUDA_VISIBLE_DEVICES=3 screen -S AirQuality_163 -d -m python TrainingModel.py --size 8000 --batch_size 32 --model_name AirQuality_163 -hs 16 --wandb -m 3 --epochs 500 --patience 50`
exec `CUDA_VISIBLE_DEVICES=3 screen -S AirQuality_LargeHSmallM -d -m python TrainingModel.py --size 8000 --batch_size 32 --model_name AirQuality_LargeHSmallM -hs 32 --wandb -m 1 --epochs 500 --patience 50`
exec `CUDA_VISIBLE_DEVICES=3 screen -S AirQuality_LargeHLargeM -d -m python TrainingModel.py --size 8000 --batch_size 32 -lr 1e-5 --model_name AirQuality_LargeHLargeM -hs 32 --wandb -m 32 --epochs 500 --patience 50 --teacher_forcing`

# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBaySmall -d -m python TrainingModel.py --size 8000 --batch_size 64 --model_name PemsBaySmall --wandb -d PemsBay -m 8 -hs 32 --epochs 500 --patience 50`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBay_Small -d -m python TrainingModel.py --size 1000 --model_name PemsBay_Small --wandb -w uniform -d PemsBay`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S WeightedPemsBay_Large -d -m python TrainingModel.py --size 5000 --model_name WeightedPemsBay_Large --wandb -d PemsBay`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBay_Large -d -m python TrainingModel.py --size 5000 --model_name PemsBay_Large --wandb -w uniform -d PemsBay`
=======
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GGTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=GGTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_SGGTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=SGGTM dataset=AirQuality`
# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_ASGGTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=ASGGTM dataset=AirQuality`

>>>>>>> GTM2



# DONE
# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=GTM dataset=AirQuality`


# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_PAR -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=PAR dataset=AirQuality`
# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_DGAN -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=DGAN dataset=AirQuality`
# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GaussianRegressor -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=GaussianRegressor dataset=AirQuality`