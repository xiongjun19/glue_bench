task_name="wsc"
model_name=$1
lr=$2
out_task="wsc"
out_dir="../data/output/${out_task}/${model_name}/${model_name}_wsc.pt"
max_seq_len=128
batch_size=16
epochs=20
accum_steps=2
train_path="../data/wsc/train.jsonl"
valid_path="../data/wsc/val.jsonl"

python ../glue_bench/run_wsc.py --pretrain_name ${model_name} --hidden_dropout 0.2  --max_seq_len ${max_seq_len} --batch_size ${batch_size} --lr $lr --epochs ${epochs} --weight_decay 1e-5 --num_workers 4 --model_path ${out_dir} --train_path ${train_path} --valid_path ${valid_path}
