set -ex
GPU_ID=0
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ../src/motion_transition.py \
	--dataroot ../dataset/lafan/test_set \
	--dataset_mode lafan \
	--name dmt \
	--model tft \
	--checkpoints_dir ../results \
	--num_joints 22 \
	--hidden_size 128 \
	--batch_size 1 \
	--epoch 100 \
	--lafan_mode seq \
	--lafan_window 11 \
	--lafan_offset 1 \
	--lafan_samplerate 1 \
	--output_path ../res/dmt_test