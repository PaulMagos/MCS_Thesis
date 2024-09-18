exec `CUDA_VISIBLE_DEVICES=2 screen -S AirQuality_SmallSeq12 -d -m python TrainingModel.py --size 8000 --batch_size 32 -lr 1e-5 --model_name AirQuality_SmallSeq4 -hs 16 --wandb -m 5 --epochs 500 --patience 50 --seq_length 4 --teacher_forcing`
exec `CUDA_VISIBLE_DEVICES=3 screen -S AirQuality_163 -d -m python TrainingModel.py --size 8000 --batch_size 32 --model_name AirQuality_163 -hs 16 --wandb -m 3 --epochs 500 --patience 50`
exec `CUDA_VISIBLE_DEVICES=3 screen -S AirQuality_LargeHSmallM -d -m python TrainingModel.py --size 8000 --batch_size 32 --model_name AirQuality_LargeHSmallM -hs 32 --wandb -m 1 --epochs 500 --patience 50`
exec `CUDA_VISIBLE_DEVICES=3 screen -S AirQuality_LargeHLargeM -d -m python TrainingModel.py --size 8000 --batch_size 32 -lr 1e-5 --model_name AirQuality_LargeHLargeM -hs 32 --wandb -m 32 --epochs 500 --patience 50 --teacher_forcing`

# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBaySmall -d -m python TrainingModel.py --size 8000 --batch_size 64 --model_name PemsBaySmall --wandb -d PemsBay -m 8 -hs 32 --epochs 500 --patience 50`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBay_Small -d -m python TrainingModel.py --size 1000 --model_name PemsBay_Small --wandb -w uniform -d PemsBay`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S WeightedPemsBay_Large -d -m python TrainingModel.py --size 5000 --model_name WeightedPemsBay_Large --wandb -d PemsBay`
# exec `CUDA_VISIBLE_DEVICES=3 screen -S PemsBay_Large -d -m python TrainingModel.py --size 5000 --model_name PemsBay_Large --wandb -w uniform -d PemsBay`

# exec `CUDA_VISIBLE_DEVICES=1 screen -S WeightedMetrLA_Small -d -m python TrainingModel.py --size 1000 --model_name WeightedMetrLA_Small --wandb -d MetrLA`
# exec `CUDA_VISIBLE_DEVICES=1 screen -S MetrLA_Small -d -m python TrainingModel.py --size 1000 --model_name MetrLA_Small --wandb -w uniform -d MetrLA`
# exec `CUDA_VISIBLE_DEVICES=1 screen -S WeightedMetrLA_Large -d -m python TrainingModel.py --size 5000 --model_name WeightedMetrLA_Large --wandb -d MetrLA`
# exec `CUDA_VISIBLE_DEVICES=1 screen -S MetrLA_Large -d -m python TrainingModel.py --size 5000 --model_name MetrLA_Large --wandb -w uniform -d MetrLA`