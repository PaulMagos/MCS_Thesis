exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=GTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GGTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=GGTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_SGGTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=SGGTM dataset=AirQuality`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_ASGGTM -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=ASGGTM dataset=AirQuality`
# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_PAR -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=PAR dataset=AirQuality`
# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_DGAN -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=DGAN dataset=AirQuality`
# exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_GaussianRegressor -d -m /data/p.magos/.pyenv/versions/3.10.14/envs/OLDE/bin/python TrainScript.py model=GaussianRegressor dataset=AirQuality`