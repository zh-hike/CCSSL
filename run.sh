# export CUDA_VISIBLE_DEVICES=6,7

python -m paddle.distributed.launch --gpus='0,1' run.py --dist

# python -u run.py
