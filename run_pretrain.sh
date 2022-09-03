conda activate torch
python main.py --config=./exps/5samples_5steps_pretrain/podnet.json
python main.py --config=./exps/5samples_5steps_pretrain/coil.json
python main.py --config=./exps/5samples_5steps_pretrain/der.json
python main.py --config=./exps/5samples_5steps_pretrain/lwf.json
python main.py --config=./exps/5samples_5steps_pretrain/icarl.json
python main.py --config=./exps/5samples_5steps_pretrain/ucir.json


python main.py --config=./exps/5samples_10steps_pretrain/podnet.json
python main.py --config=./exps/5samples_10steps_pretrain/coil.json
python main.py --config=./exps/5samples_10steps_pretrain/der.json
python main.py --config=./exps/5samples_10steps_pretrain/lwf.json
python main.py --config=./exps/5samples_10steps_pretrain/icarl.json
python main.py --config=./exps/5samples_10steps_pretrain/ucir.json



#CUDA_VISIBLE_DEVICES=4,5,6,7