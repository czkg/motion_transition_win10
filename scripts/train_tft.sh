set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/train.py \
	--dataroot ../dataset/lafan/train_set \
	--dataset_mode lafan \
	--name tft \
	--model tft \
	--checkpoints_dir ../results \
	--niter 70 \
	--niter_decay 30 \
	--num_joints 22 \
	--hidden_size 128 \
	--lr 1e-4 \
	--beta1 0.9 \
	--init_type xavier \
	--init_gain 0.8 \
	--batch_size 32 \
	--past_len 10 \
	--lafan_mode seq \
	--lafan_window 11 \
	--lafan_offset 1 \
	--lafan_samplerate 1