exec `CUDA_VISIBLE_DEVICES=3 screen -S Synthetic_GTM -d -m python TrainScript.py model=GTM dataset=Synthetic`
exec `CUDA_VISIBLE_DEVICES=3 screen -S Synthetic_GGTM -d -m python TrainScript.py model=GGTM dataset=Synthetic`
exec `CUDA_VISIBLE_DEVICES=3 screen -S Synthetic_ASGGTM -d -m python TrainScript.py model=ASGGTM dataset=Synthetic`
exec `CUDA_VISIBLE_DEVICES=3 screen -S Synthetic_PAR -d -m python TrainScript.py model=PAR dataset=Synthetic`
exec `CUDA_VISIBLE_DEVICES=3 screen -S Synthetic_DGAN -d -m python TrainScript.py model=DGAN dataset=Synthetic`
exec `CUDA_VISIBLE_DEVICES=3 screen -S Synthetic_GaussianRegressor -d -m python TrainScript.py model=GaussianRegressor dataset=Synthetic`