#!/usr/bin/env bash

set -e
set -x
# model_name_arr=("bert-base-cased" "bert-large-cased" "albert-base-v2" "albert-large-v2" "albert-xlarge-v2" "roberta-base" "roberta-large")
# lr_arr=(2e-5 2e-5 2e-5 2e-5 1e-5 2e-5 1e-5)
model_name_arr=("albert-large-v2" "albert-xlarge-v2")
lr_arr=(8e-6 4e-6)


for(( i=0;i<${#model_name_arr[@]};i++)) do
	bash run_wsc_albert.sh ${model_name_arr[i]} ${lr_arr[i]};
done

