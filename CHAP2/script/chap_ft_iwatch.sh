#!/bin/bash
# Print the current directory
pip install transformers

echo "Current directory: $(pwd)"

# List all files in the current directory
echo "Files:"
ls -lah

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CHAP \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 1.0 \
--pos_weight 1.0 \
--use_data_aug 1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CHAP \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 1.0 \
--pos_weight 1.0 \
--use_data_aug 1 


echo "All tasks completed."

##
# chmod +x script/chap_iwatch.sh
# ./script/chap_iwatch.sh