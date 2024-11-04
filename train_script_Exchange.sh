exec `CUDA_VISIBLE_DEVICES=1 screen -S Exchange_GTM -d -m python TrainScript.py model=GTM dataset=Exchange`
exec `CUDA_VISIBLE_DEVICES=1 screen -S Exchange_GGTM -d -m python TrainScript.py model=GGTM dataset=Exchange`
exec `CUDA_VISIBLE_DEVICES=1 screen -S Exchange_ASGGTM -d -m python TrainScript.py model=ASGGTM dataset=Exchange`
exec `CUDA_VISIBLE_DEVICES=1 screen -S Exchange_PAR -d -m python TrainScript.py model=PAR dataset=Exchange`
exec `CUDA_VISIBLE_DEVICES=1 screen -S Exchange_DGAN -d -m python TrainScript.py model=DGAN dataset=Exchange`
exec `CUDA_VISIBLE_DEVICES=1 screen -S Exchange_GaussianRegressor -d -m python TrainScript.py model=GaussianRegressor dataset=Exchange`