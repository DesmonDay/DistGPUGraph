source activate py36

sh kill_all.sh
NCCL_HOME=/root/paddlejob/nccl_2.8.3-1+cuda10.1_x86_64/
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1 python train_gcn.py --hidden_size 256 --dataset reddit --num_layers 1 --epoch 200 --mode random
# CUDA_VISIBLE_DEVICES=0,1 python train_gcn.py --hidden_size 256 --dataset reddit --num_layers 1 --epoch 200 --mode metis
# CUDA_VISIBLE_DEVICES=0,1, python train_gcn.py --hidden_size 256 --dataset reddit --num_layers 1 --epoch 200 --mode metis_w
