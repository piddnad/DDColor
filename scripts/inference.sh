CUDA_VISIBLE_DEVICES=0 \
python3 inference/colorization_pipline.py \
	--input ./test_imgs --output ./colorize_output \
	--model_path pretrain/pytorch_model.pt