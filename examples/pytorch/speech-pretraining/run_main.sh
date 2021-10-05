#!/usr/bin/env bash
#accelerate launch run_wav2vec2_pretraining_no_trainer.py \
python run_wav2vec2_pretraining_no_trainer.py \
        --dataset_name="librispeech_asr" \
        --dataset_config_names clean clean \
				--dataset_split_names validation test \
        --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
        --output_dir="./wav2vec2-pretrained-demo" \
        --max_train_steps="200000" \
        --num_warmup_steps="32000" \
        --gradient_accumulation_steps="4" \
				--gradient_checkpointing \
        --learning_rate="0.001" \
        --weight_decay="0.01" \
        --max_duration_in_seconds="15.0" \
        --min_duration_in_seconds="2.0" \
        --logging_steps="1" \
        --saving_steps="10000" \
        --per_device_train_batch_size="8" \
        --per_device_eval_batch_size="8" \
        --adam_beta1="0.9" \
        --adam_beta2="0.98" \
        --adam_epsilon="1e-06" \
				--push_to_hub \
				--preprocessing_only \
