conda activate torch
python main.py --config=./exps/5samples_5steps/podnet.json
python main.py --config=./exps/5samples_5steps/coil.json
python main.py --config=./exps/5samples_5steps/der.json
python main.py --config=./exps/5samples_5steps/lwf.json
python main.py --config=./exps/5samples_5steps/icarl.json
python main.py --config=./exps/5samples_5steps/ucir.json
python main.py --config=./exps/5samples_10steps/podnet.json
python main.py --config=./exps/5samples_10steps/coil.json
python main.py --config=./exps/5samples_10steps/der.json
python main.py --config=./exps/5samples_10steps/lwf.json
python main.py --config=./exps/5samples_10steps/icarl.json
python main.py --config=./exps/5samples_10steps/ucir.json


#CUDA_VISIBLE_DEVICES=4,5,6,7
#CUDA_VISIBLE_DEVICES=3,4,5,6
