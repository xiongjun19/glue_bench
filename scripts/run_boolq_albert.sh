set -e 
set -x

export WANDB_DISABLED=True

# model_name="roberta-base"
model_name='albert-xlarge-v2'
out_task="boolq"
out_dir="../data/output/${out_task}/${model_name}"
max_seq_len=128
batch_size=16
lr=4e-6
epochs=3
accum_steps=2


train_path="../data/boolq/train.json"
val_path="../data/boolq/val.json"

python ../glue_bench/run_glue.py --model_name_or_path ${model_name}  --do_train --do_eval --max_seq_length ${max_seq_len} --per_device_train_batch_size ${batch_size} --learning_rate $lr --num_train_epochs ${epochs} --output_dir ${out_dir} --train_file ${train_path} --validation_file ${val_path} --gradient_accumulation_steps ${accum_steps} 
