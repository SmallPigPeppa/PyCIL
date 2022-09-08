conda activate torch

#/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_5steps/podnet.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_5steps/coil.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_5steps/der.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_5steps/lwf.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_5steps/icarl.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_5steps/ucir.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_10steps/podnet.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_10steps/coil.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_10steps/der.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_10steps/lwf.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_10steps/icarl.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_10steps/ucir.json

/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_5steps/coil.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/5samples_10steps/coil.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/10samples_5steps/coil.json
/root/miniconda3/envs/torch/bin/python main.py --config=./exps/10samples_10steps/coil.json
#CUDA_VISIBLE_DEVICES=4,5,6,7