CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=3721 basicsr/train.py \
    -opt options/train/train_ddcolor.yml --auto_resume --launcher pytorch
