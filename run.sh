deepspeed train_gptj_toolformer.py --model_name_or_path=EleutherAI/gpt-j-6B --per_device_train_batch_size=1 \
  --num_train_epochs 10 --save_strategy=epoch --output_dir=finetune_toolformer_v0 --report_to "wandb" \
  --dataset_name dmayhem93/toolformer-v0-postprocessed --tokenizer_name customToolformer \
  --block_size 2048 --gradient_accumulation_steps 1 --do_train --do_eval --evaluation_strategy=epoch \
  --logging_strategy=epoch --fp16 --overwrite_output_dir --adam_beta1=0.9 --adam_beta2=0.999 \
  --weight_decay=2e-02 --learning_rate=1e-05 --warmup_steps=100 --per_device_eval_batch_size=1 \
  --cache_dir="hf_cache" --gradient_checkpointing=True --deepspeed ds_config_gpt_j.json