torchrun --nproc_per_node=1 -m main_finetune \
--data_path "/niddk-data-central/SOL/PASOS/train/SOL_10hz" \
--model CHAP \
--eval "CHAP2/SUBMIT_RESULT/CHAP_FT_SOL/checkpoint-submit.pth" \
--make_prediction \
--prediction_dir "/niddk-data-central/sol_predictions" \
--batch_size 16
