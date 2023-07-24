CUDA_VISIBLE_DEVICES=0,1 bash run_10samples_5steps.sh &
CUDA_VISIBLE_DEVICES=2,3 bash run_10samples_5steps_pretrain.sh &
CUDA_VISIBLE_DEVICES=4,5 bash run_10samples_10steps.sh &
CUDA_VISIBLE_DEVICES=6,7 bash run_10samples_10steps_pretrain.sh &
