exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GTM -d -m python TrainScript.py model=GTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GGTM -d -m python TrainScript.py model=GGTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_SGGTM -d -m python TrainScript.py model=SGGTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_ASGGTM -d -m python TrainScript.py model=ASGGTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_PAR -d -m python TrainScript.py model=PAR dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_DGAN -d -m python TrainScript.py model=DGAN dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GaussianRegressor -d -m python TrainScript.py model=GaussianRegressor dataset=AirQuality`