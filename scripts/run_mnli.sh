export WANDB_DISABLED=True

task_name="mnli"
model_name=$1
lr=$2
out_task="mnli"
out_dir="../data/output/${out_task}/${model_name}"
max_seq_len=128
batch_size=16
epochs=3
accum_steps=2

python ../glue_bench/run_glue.py --model_name_or_path ${model_name} --task_name ${task_name} --do_train --do_eval --max_seq_length ${max_seq_len} --per_device_train_batch_size ${batch_size} --learning_rate $lr --num_train_epochs ${epochs} --output_dir ${out_dir} --gradient_accumulation_steps ${accum_steps} --save_total_limit 5 --overwrite_output_dir
