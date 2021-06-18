#!/usr/bin/env bash

set -e
set -x
model_name_arr=("bert-base-cased" "albert-base-v2" "roberta-base" "roberta-large")
# model_name_arr=("bert-large-cased" "roberta-base" "roberta-large")

for(( i=0;i<${#model_name_arr[@]};i++)) do
	bash run_boolq.sh ${model_name_arr[i]};
done


