exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_Small -d -m python TrainingModel.py --size 1000 --model_name AirQuality_Small --wandb -m 2`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_Medium -d -m python TrainingModel.py --size 5000 --model_name AirQuality_Medium --wandb -m 3`
exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_Large -d -m python TrainingModel.py --size 8000 --model_name AirQuality_Large --wandb -m 3`

# exec `CUDA_VISIBLE_DEVICES=3 screen -S WeightedPemsBay_Small -d -m python TrainingModel.py --size 1000 --model_name WeightedPemsBay_Small --wandb -d PemsBay`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBay_Small -d -m python TrainingModel.py --size 1000 --model_name PemsBay_Small --wandb -w uniform -d PemsBay`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S WeightedPemsBay_Large -d -m python TrainingModel.py --size 5000 --model_name WeightedPemsBay_Large --wandb -d PemsBay`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBay_Large -d -m python TrainingModel.py --size 5000 --model_name PemsBay_Large --wandb -w uniform -d PemsBay`

# exec `CUDA_VISIBLE_DEVICES=1 screen -S WeightedMetrLA_Small -d -m python TrainingModel.py --size 1000 --model_name WeightedMetrLA_Small --wandb -d MetrLA`
# exec `CUDA_VISIBLE_DEVICES=1 screen -S MetrLA_Small -d -m python TrainingModel.py --size 1000 --model_name MetrLA_Small --wandb -w uniform -d MetrLA`
# exec `CUDA_VISIBLE_DEVICES=1 screen -S WeightedMetrLA_Large -d -m python TrainingModel.py --size 5000 --model_name WeightedMetrLA_Large --wandb -d MetrLA`
# exec `CUDA_VISIBLE_DEVICES=1 screen -S MetrLA_Large -d -m python TrainingModel.py --size 5000 --model_name MetrLA_Large --wandb -w uniform -d MetrLA`