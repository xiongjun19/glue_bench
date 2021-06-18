#!/usr/bin/env bash

set -e
set -x
model_name_arr=("bert-base-cased" "bert-large-cased" "albert-base-v2" "albert-large-v2" "roberta-base" "roberta-large")
lr_arr=(2e-5 8e-6 2e-5 8e-6 2e-5 1e-5)

for(( i=0;i<${#model_name_arr[@]};i++)) do
	bash run_mnli.sh ${model_name_arr[i]} ${lr_arr[i]};
done


