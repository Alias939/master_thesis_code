All the operations on the datasets before the finetuning process were done in jupyter notebook. The file is pre_finetune.ipynb. 

The finetuning process is ran through the commmand:

train.py --languages nl --output_dir output/1024 --do_train --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --distributed_softmax --max_steps 224 --evaluation_strategy no --eval_steps 0 --max_seq_len 264 --warmup_steps 0 --label_names page_id --logging_steps 1 --metric_for_best_model eval_global_mrr --load_best_model_at_end False --save_total_limit 3 --report_to tensorboard --dataloader_num_workers 1 --single_domain --hidden_dropout_prob 0 --learning_rate 0.00001 --weight_decay 0.01 --alpha 1 --gradient_checkpointing False
