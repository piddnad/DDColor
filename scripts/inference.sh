CUDA_VISIBLE_DEVICES=0 \
python3 infer.py \
	--input ./assets/test_images --output ./colorize_output \
	--model_path modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt