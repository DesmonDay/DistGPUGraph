source activate py36

sh kill_all.sh
NCCL_HOME=/root/paddlejob/nccl_2.8.3-1+cuda10.1_x86_64/
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1 python dist_train.py --conf config/gcn.yaml --dataset reddit --epoch 200
