+ model_name_arr=("bert-base-cased" "bert-large-cased" "albert-base-v2" "albert-large-v2" "albert-xlarge-v2" "roberta-base" "roberta-large")
+ (( i=0 ))
+ (( i<7 ))
+ bash run_boolq.sh bert-base-cased
+ export WANDB_DISABLED=True
+ WANDB_DISABLED=True
+ model_name=bert-base-cased
+ out_task=boolq
+ out_dir=../data/output/boolq/bert-base-cased
+ max_seq_len=128
+ batch_size=16
+ lr=4e-5
+ epochs=40
+ accum_steps=8
+ train_path=../data/boolq/train.json
+ val_path=../data/boolq/val.json
+ python ../glue_bench/run_glue.py --model_name_or_path bert-base-cased --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 4e-5 --num_train_epochs 40 --output_dir ../data/output/boolq/bert-base-cased --train_file ../data/boolq/train.json --validation_file ../data/boolq/val.json --gradient_accumulation_steps 8 --overwrite_output_dir
Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
06/10/2021 13:21:34 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
06/10/2021 13:21:34 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir=../data/output/boolq/bert-base-cased, overwrite_output_dir=True, do_train=True, do_eval=True, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=8, gradient_accumulation_steps=8, eval_accumulation_steps=None, learning_rate=4e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=40.0, max_steps=-1, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=0, logging_dir=runs/Jun10_13-21-34_ip-172-31-14-159, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=500, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name=../data/output/boolq/bert-base-cased, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, length_column_name=length, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, _n_gpu=1, mp_parameters=)
06/10/2021 13:21:34 - INFO - __main__ -   load a local file for train: ../data/boolq/train.json
06/10/2021 13:21:34 - INFO - __main__ -   load a local file for validation: ../data/boolq/val.json
Using custom data configuration default
Reusing dataset json (/home/ubuntu/.cache/huggingface/datasets/json/default-87e14794c70b068f/0.0.0/fb88b12bd94767cb0cc7eedcd82ea1f402d2162addc03a37e81d4f8dc7313ad9)
[INFO|file_utils.py:1426] 2021-06-10 13:21:35,553 >> https://huggingface.co/bert-base-cased/resolve/main/config.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpjpjcepnz

[INFO|file_utils.py:1430] 2021-06-10 13:21:35,883 >> storing https://huggingface.co/bert-base-cased/resolve/main/config.json in cache at /home/ubuntu/.cache/huggingface/transformers/a803e0468a8fe090683bdc453f4fac622804f49de86d7cecaee92365d4a0f829.a64a22196690e0e82ead56f388a3ef3a50de93335926ccfa20610217db589307
[INFO|file_utils.py:1433] 2021-06-10 13:21:35,883 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/a803e0468a8fe090683bdc453f4fac622804f49de86d7cecaee92365d4a0f829.a64a22196690e0e82ead56f388a3ef3a50de93335926ccfa20610217db589307
[INFO|configuration_utils.py:491] 2021-06-10 13:21:35,884 >> loading configuration file https://huggingface.co/bert-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/a803e0468a8fe090683bdc453f4fac622804f49de86d7cecaee92365d4a0f829.a64a22196690e0e82ead56f388a3ef3a50de93335926ccfa20610217db589307
[INFO|configuration_utils.py:527] 2021-06-10 13:21:35,884 >> Model config BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 28996
}

[INFO|configuration_utils.py:491] 2021-06-10 13:21:36,205 >> loading configuration file https://huggingface.co/bert-base-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/a803e0468a8fe090683bdc453f4fac622804f49de86d7cecaee92365d4a0f829.a64a22196690e0e82ead56f388a3ef3a50de93335926ccfa20610217db589307
[INFO|configuration_utils.py:527] 2021-06-10 13:21:36,205 >> Model config BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 28996
}

[INFO|file_utils.py:1426] 2021-06-10 13:21:36,533 >> https://huggingface.co/bert-base-cased/resolve/main/vocab.txt not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpwxil9g3w

[INFO|file_utils.py:1430] 2021-06-10 13:21:37,015 >> storing https://huggingface.co/bert-base-cased/resolve/main/vocab.txt in cache at /home/ubuntu/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|file_utils.py:1433] 2021-06-10 13:21:37,015 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|file_utils.py:1426] 2021-06-10 13:21:37,345 >> https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpxx5qrk_k

[INFO|file_utils.py:1430] 2021-06-10 13:21:37,906 >> storing https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json in cache at /home/ubuntu/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6
[INFO|file_utils.py:1433] 2021-06-10 13:21:37,906 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6
[INFO|file_utils.py:1426] 2021-06-10 13:21:38,885 >> https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpkp7el6pd

[INFO|file_utils.py:1430] 2021-06-10 13:21:39,208 >> storing https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json in cache at /home/ubuntu/.cache/huggingface/transformers/ec84e86ee39bfe112543192cf981deebf7e6cbe8c91b8f7f8f63c9be44366158.ec5c189f89475aac7d8cbd243960a0655cfadc3d0474da8ff2ed0bf1699c2a5f
[INFO|file_utils.py:1433] 2021-06-10 13:21:39,208 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/ec84e86ee39bfe112543192cf981deebf7e6cbe8c91b8f7f8f63c9be44366158.ec5c189f89475aac7d8cbd243960a0655cfadc3d0474da8ff2ed0bf1699c2a5f
[INFO|tokenization_utils_base.py:1707] 2021-06-10 13:21:39,208 >> loading file https://huggingface.co/bert-base-cased/resolve/main/vocab.txt from cache at /home/ubuntu/.cache/huggingface/transformers/6508e60ab3c1200bffa26c95f4b58ac6b6d95fba4db1f195f632fa3cd7bc64cc.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|tokenization_utils_base.py:1707] 2021-06-10 13:21:39,209 >> loading file https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json from cache at /home/ubuntu/.cache/huggingface/transformers/226a307193a9f4344264cdc76a12988448a25345ba172f2c7421f3b6810fddad.3dab63143af66769bbb35e3811f75f7e16b2320e12b7935e216bd6159ce6d9a6
[INFO|tokenization_utils_base.py:1707] 2021-06-10 13:21:39,209 >> loading file https://huggingface.co/bert-base-cased/resolve/main/added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 13:21:39,209 >> loading file https://huggingface.co/bert-base-cased/resolve/main/special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 13:21:39,209 >> loading file https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json from cache at /home/ubuntu/.cache/huggingface/transformers/ec84e86ee39bfe112543192cf981deebf7e6cbe8c91b8f7f8f63c9be44366158.ec5c189f89475aac7d8cbd243960a0655cfadc3d0474da8ff2ed0bf1699c2a5f
[INFO|file_utils.py:1426] 2021-06-10 13:21:39,562 >> https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpmw183f0i

[INFO|file_utils.py:1430] 2021-06-10 13:21:55,317 >> storing https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin in cache at /home/ubuntu/.cache/huggingface/transformers/092cc582560fc3833e556b3f833695c26343cb54b7e88cd02d40821462a74999.1f48cab6c959fc6c360d22bea39d06959e90f5b002e77e836d2da45464875cda
[INFO|file_utils.py:1433] 2021-06-10 13:21:55,317 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/092cc582560fc3833e556b3f833695c26343cb54b7e88cd02d40821462a74999.1f48cab6c959fc6c360d22bea39d06959e90f5b002e77e836d2da45464875cda
[INFO|modeling_utils.py:1052] 2021-06-10 13:21:55,319 >> loading weights file https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/092cc582560fc3833e556b3f833695c26343cb54b7e88cd02d40821462a74999.1f48cab6c959fc6c360d22bea39d06959e90f5b002e77e836d2da45464875cda
[WARNING|modeling_utils.py:1160] 2021-06-10 13:21:59,735 >> Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1171] 2021-06-10 13:21:59,735 >> Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


06/10/2021 13:22:01 - INFO - __main__ -   Sample 409 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'idx': 409, 'input_ids': [101, 1169, 1128, 1855, 1103, 1171, 4015, 1113, 170, 1714, 4932, 102, 4299, 4932, 118, 118, 4299, 11784, 1169, 5156, 1129, 2046, 1120, 170, 1344, 6556, 1118, 1363, 2139, 119, 1130, 1103, 5803, 117, 1211, 2139, 1294, 3102, 118, 118, 2908, 110, 1104, 1147, 4021, 119, 1109, 2074, 112, 188, 1436, 14896, 1116, 113, 1216, 1112, 3036, 10544, 117, 5512, 5631, 117, 4150, 4522, 117, 5180, 21056, 4722, 117, 3620, 20422, 117, 19343, 3902, 117, 4101, 12786, 6922, 117, 1105, 19862, 1986, 8976, 2293, 114, 1169, 1294, 4986, 3078, 110, 1104, 1147, 4021, 1166, 170, 1265, 117, 1229, 14140, 1193, 2869, 14896, 1116, 113, 174, 119, 176, 119, 14868, 4115, 117, 3177, 1592, 3276, 1874, 4421, 117, 160, 14080, 15506, 117, 12971, 20978, 117, 1262, 4889, 139, 102], 'label': 1, 'passage': "Free throw -- Free throws can normally be shot at a high percentage by good players. In the NBA, most players make 70--80% of their attempts. The league's best shooters (such as Steve Nash, Rick Barry, Ray Allen, José Calderón, Stephen Curry, Reggie Miller, Kevin Durant, and Dirk Nowitzki) can make roughly 90% of their attempts over a season, while notoriously poor shooters (e.g. Dwight Howard, DeAndre Jordan, Wilt Chamberlain, Andre Drummond, Andris Biedrins, Chris Dudley, Ben Wallace, Shaquille O'Neal, and Dennis Rodman) may struggle to make 50% of them. During a foul shot, a player's foot must be completely behind the foul line. If a player lines up with part of his or her foot on the line, a violation is called and the shot does not count. Foul shots are worth one point.", 'question': 'can you hit the backboard on a free throw', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}.
06/10/2021 13:22:01 - INFO - __main__ -   Sample 4506 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 4506, 'input_ids': [101, 1169, 178, 3952, 170, 2998, 1107, 1103, 6346, 1443, 170, 1862, 4134, 102, 11121, 4134, 118, 118, 1109, 1862, 4134, 1110, 1136, 2320, 1113, 14889, 6346, 119, 1438, 117, 2960, 1104, 170, 1862, 4134, 17356, 1103, 14889, 1555, 1121, 1217, 1682, 1106, 1862, 1103, 8926, 1191, 1122, 17617, 5576, 21091, 4121, 1895, 132, 1216, 1112, 1121, 3290, 117, 2112, 2553, 1496, 117, 1137, 22475, 7680, 119, 5723, 6346, 1336, 4303, 1561, 2044, 2998, 6346, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 1, 'passage': 'Return address -- The return address is not required on postal mail. However, lack of a return address prevents the postal service from being able to return the item if it proves undeliverable; such as from damage, postage due, or invalid destination. Such mail may otherwise become dead letter mail.', 'question': 'can i send a letter in the mail without a return address', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.

[INFO|trainer.py:491] 2021-06-10 13:22:05,412 >> The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: passage, idx, question.
[INFO|trainer.py:491] 2021-06-10 13:22:05,412 >> The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: passage, idx, question.
[INFO|trainer.py:1013] 2021-06-10 13:22:05,496 >> ***** Running training *****
[INFO|trainer.py:1014] 2021-06-10 13:22:05,496 >>   Num examples = 9427
[INFO|trainer.py:1015] 2021-06-10 13:22:05,496 >>   Num Epochs = 40
[INFO|trainer.py:1016] 2021-06-10 13:22:05,496 >>   Instantaneous batch size per device = 16
[INFO|trainer.py:1017] 2021-06-10 13:22:05,496 >>   Total train batch size (w. parallel, distributed & accumulation) = 128
[INFO|trainer.py:1018] 2021-06-10 13:22:05,496 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:1019] 2021-06-10 13:22:05,496 >>   Total optimization steps = 2920

[INFO|configuration_utils.py:329] 2021-06-10 13:36:48,900 >> Configuration saved in ../data/output/boolq/bert-base-cased/checkpoint-500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 13:36:49,886 >> Model weights saved in ../data/output/boolq/bert-base-cased/checkpoint-500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 13:36:49,887 >> tokenizer config file saved in ../data/output/boolq/bert-base-cased/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 13:36:49,887 >> Special tokens file saved in ../data/output/boolq/bert-base-cased/checkpoint-500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 13:52:56,234 >> Configuration saved in ../data/output/boolq/bert-base-cased/checkpoint-1000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 13:52:57,210 >> Model weights saved in ../data/output/boolq/bert-base-cased/checkpoint-1000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 13:52:57,211 >> tokenizer config file saved in ../data/output/boolq/bert-base-cased/checkpoint-1000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 13:52:57,211 >> Special tokens file saved in ../data/output/boolq/bert-base-cased/checkpoint-1000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 14:08:50,076 >> Configuration saved in ../data/output/boolq/bert-base-cased/checkpoint-1500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 14:08:51,070 >> Model weights saved in ../data/output/boolq/bert-base-cased/checkpoint-1500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 14:08:51,071 >> tokenizer config file saved in ../data/output/boolq/bert-base-cased/checkpoint-1500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 14:08:51,071 >> Special tokens file saved in ../data/output/boolq/bert-base-cased/checkpoint-1500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 14:24:40,683 >> Configuration saved in ../data/output/boolq/bert-base-cased/checkpoint-2000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 14:24:41,655 >> Model weights saved in ../data/output/boolq/bert-base-cased/checkpoint-2000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 14:24:41,656 >> tokenizer config file saved in ../data/output/boolq/bert-base-cased/checkpoint-2000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 14:24:41,656 >> Special tokens file saved in ../data/output/boolq/bert-base-cased/checkpoint-2000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 14:40:33,061 >> Configuration saved in ../data/output/boolq/bert-base-cased/checkpoint-2500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 14:40:34,033 >> Model weights saved in ../data/output/boolq/bert-base-cased/checkpoint-2500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 14:40:34,033 >> tokenizer config file saved in ../data/output/boolq/bert-base-cased/checkpoint-2500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 14:40:34,034 >> Special tokens file saved in ../data/output/boolq/bert-base-cased/checkpoint-2500/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)



[INFO|trainer.py:1648] 2021-06-10 14:53:50,208 >> Saving model checkpoint to ../data/output/boolq/bert-base-cased
[INFO|configuration_utils.py:329] 2021-06-10 14:53:50,209 >> Configuration saved in ../data/output/boolq/bert-base-cased/config.json
[INFO|modeling_utils.py:831] 2021-06-10 14:53:51,308 >> Model weights saved in ../data/output/boolq/bert-base-cased/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 14:53:51,308 >> tokenizer config file saved in ../data/output/boolq/bert-base-cased/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 14:53:51,308 >> Special tokens file saved in ../data/output/boolq/bert-base-cased/special_tokens_map.json
[INFO|trainer_pt_utils.py:722] 2021-06-10 14:53:51,347 >> ***** train metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   epoch                      =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   init_mem_cpu_alloc_delta   =     2088MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   init_mem_cpu_peaked_delta  =      407MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   init_mem_gpu_alloc_delta   =      413MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   init_mem_gpu_peaked_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   train_mem_cpu_alloc_delta  =       96MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   train_mem_cpu_peaked_delta =      255MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   train_mem_gpu_alloc_delta  =     1266MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   train_mem_gpu_peaked_delta =     1706MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   train_runtime              = 1:31:44.60
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   train_samples              =       9427
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:53:51,347 >>   train_samples_per_second   =       0.53
{'loss': 0.3054, 'learning_rate': 3.315068493150685e-05, 'epoch': 6.84}
{'loss': 0.0387, 'learning_rate': 2.6301369863013698e-05, 'epoch': 13.69}
{'loss': 0.0139, 'learning_rate': 1.945205479452055e-05, 'epoch': 20.54}
{'loss': 0.0042, 'learning_rate': 1.2602739726027398e-05, 'epoch': 27.39}
{'loss': 0.0019, 'learning_rate': 5.753424657534246e-06, 'epoch': 34.24}
{'train_runtime': 5504.6091, 'train_samples_per_second': 0.53, 'epoch': 39.99}
06/10/2021 14:53:51 - INFO - __main__ -   *** Evaluate ***
[INFO|trainer.py:491] 2021-06-10 14:53:51,393 >> The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: passage, idx, question.
[INFO|trainer.py:1865] 2021-06-10 14:53:51,394 >> ***** Running Evaluation *****
[INFO|trainer.py:1866] 2021-06-10 14:53:51,394 >>   Num examples = 3270
[INFO|trainer.py:1867] 2021-06-10 14:53:51,394 >>   Batch size = 8

[INFO|trainer_pt_utils.py:722] 2021-06-10 14:54:15,929 >> ***** eval metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   epoch                     =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_accuracy             =     0.7394
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_loss                 =     2.1707
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_mem_cpu_alloc_delta  =        9MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_mem_cpu_peaked_delta =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_mem_gpu_alloc_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_mem_gpu_peaked_delta =       33MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_runtime              = 0:00:24.48
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_samples              =       3270
[INFO|trainer_pt_utils.py:727] 2021-06-10 14:54:15,930 >>   eval_samples_per_second   =    133.548
+ (( i++ ))
+ (( i<7 ))
+ bash run_boolq.sh bert-large-cased
+ export WANDB_DISABLED=True
+ WANDB_DISABLED=True
+ model_name=bert-large-cased
+ out_task=boolq
+ out_dir=../data/output/boolq/bert-large-cased
+ max_seq_len=128
+ batch_size=16
+ lr=4e-5
+ epochs=40
+ accum_steps=8
+ train_path=../data/boolq/train.json
+ val_path=../data/boolq/val.json
+ python ../glue_bench/run_glue.py --model_name_or_path bert-large-cased --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 4e-5 --num_train_epochs 40 --output_dir ../data/output/boolq/bert-large-cased --train_file ../data/boolq/train.json --validation_file ../data/boolq/val.json --gradient_accumulation_steps 8 --overwrite_output_dir
Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
06/10/2021 14:54:18 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
06/10/2021 14:54:18 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir=../data/output/boolq/bert-large-cased, overwrite_output_dir=True, do_train=True, do_eval=True, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=8, gradient_accumulation_steps=8, eval_accumulation_steps=None, learning_rate=4e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=40.0, max_steps=-1, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=0, logging_dir=runs/Jun10_14-54-18_ip-172-31-14-159, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=500, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name=../data/output/boolq/bert-large-cased, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, length_column_name=length, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, _n_gpu=1, mp_parameters=)
06/10/2021 14:54:18 - INFO - __main__ -   load a local file for train: ../data/boolq/train.json
06/10/2021 14:54:18 - INFO - __main__ -   load a local file for validation: ../data/boolq/val.json
Using custom data configuration default
Reusing dataset json (/home/ubuntu/.cache/huggingface/datasets/json/default-87e14794c70b068f/0.0.0/fb88b12bd94767cb0cc7eedcd82ea1f402d2162addc03a37e81d4f8dc7313ad9)
[INFO|file_utils.py:1426] 2021-06-10 14:54:19,884 >> https://huggingface.co/bert-large-cased/resolve/main/config.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmp263gxpdr

[INFO|file_utils.py:1430] 2021-06-10 14:54:20,217 >> storing https://huggingface.co/bert-large-cased/resolve/main/config.json in cache at /home/ubuntu/.cache/huggingface/transformers/11ad22b0deaa199d15b331609ca5f60872a1a91473e9b40c115192dadb6d9a30.bdf0177a774dcff07681b2527b926c099e6563687c75a79f7469c7a7da7898c7
[INFO|file_utils.py:1433] 2021-06-10 14:54:20,217 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/11ad22b0deaa199d15b331609ca5f60872a1a91473e9b40c115192dadb6d9a30.bdf0177a774dcff07681b2527b926c099e6563687c75a79f7469c7a7da7898c7
[INFO|configuration_utils.py:491] 2021-06-10 14:54:20,217 >> loading configuration file https://huggingface.co/bert-large-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/11ad22b0deaa199d15b331609ca5f60872a1a91473e9b40c115192dadb6d9a30.bdf0177a774dcff07681b2527b926c099e6563687c75a79f7469c7a7da7898c7
[INFO|configuration_utils.py:527] 2021-06-10 14:54:20,218 >> Model config BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 28996
}

[INFO|configuration_utils.py:491] 2021-06-10 14:54:20,547 >> loading configuration file https://huggingface.co/bert-large-cased/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/11ad22b0deaa199d15b331609ca5f60872a1a91473e9b40c115192dadb6d9a30.bdf0177a774dcff07681b2527b926c099e6563687c75a79f7469c7a7da7898c7
[INFO|configuration_utils.py:527] 2021-06-10 14:54:20,548 >> Model config BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 28996
}

[INFO|file_utils.py:1426] 2021-06-10 14:54:20,873 >> https://huggingface.co/bert-large-cased/resolve/main/vocab.txt not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpmyxp2u8n

[INFO|file_utils.py:1430] 2021-06-10 14:54:21,360 >> storing https://huggingface.co/bert-large-cased/resolve/main/vocab.txt in cache at /home/ubuntu/.cache/huggingface/transformers/c9961ea5b7e8ad58701728c45f4d225f70b19aa59745121e5a96c8a44efca4c8.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|file_utils.py:1433] 2021-06-10 14:54:21,361 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/c9961ea5b7e8ad58701728c45f4d225f70b19aa59745121e5a96c8a44efca4c8.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|file_utils.py:1426] 2021-06-10 14:54:21,695 >> https://huggingface.co/bert-large-cased/resolve/main/tokenizer.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpdbi_l_s7

[INFO|file_utils.py:1430] 2021-06-10 14:54:22,185 >> storing https://huggingface.co/bert-large-cased/resolve/main/tokenizer.json in cache at /home/ubuntu/.cache/huggingface/transformers/75be22d7750034989358861e325977feda47740e1c3f8a4dc1cb73570aad843e.2b9a196704f2f183fe3f4b48d6e662dba8203fdcb3346bfa896831378edf6f97
[INFO|file_utils.py:1433] 2021-06-10 14:54:22,185 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/75be22d7750034989358861e325977feda47740e1c3f8a4dc1cb73570aad843e.2b9a196704f2f183fe3f4b48d6e662dba8203fdcb3346bfa896831378edf6f97
[INFO|file_utils.py:1426] 2021-06-10 14:54:23,164 >> https://huggingface.co/bert-large-cased/resolve/main/tokenizer_config.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpg5tnorkl

[INFO|file_utils.py:1430] 2021-06-10 14:54:23,500 >> storing https://huggingface.co/bert-large-cased/resolve/main/tokenizer_config.json in cache at /home/ubuntu/.cache/huggingface/transformers/45d2aa048795efc7b12791662c188d5e3aa2f9ac54b2cf3f6e4d7bc6544e3d13.ec5c189f89475aac7d8cbd243960a0655cfadc3d0474da8ff2ed0bf1699c2a5f
[INFO|file_utils.py:1433] 2021-06-10 14:54:23,500 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/45d2aa048795efc7b12791662c188d5e3aa2f9ac54b2cf3f6e4d7bc6544e3d13.ec5c189f89475aac7d8cbd243960a0655cfadc3d0474da8ff2ed0bf1699c2a5f
[INFO|tokenization_utils_base.py:1707] 2021-06-10 14:54:23,500 >> loading file https://huggingface.co/bert-large-cased/resolve/main/vocab.txt from cache at /home/ubuntu/.cache/huggingface/transformers/c9961ea5b7e8ad58701728c45f4d225f70b19aa59745121e5a96c8a44efca4c8.437aa611e89f6fc6675a049d2b5545390adbc617e7d655286421c191d2be2791
[INFO|tokenization_utils_base.py:1707] 2021-06-10 14:54:23,500 >> loading file https://huggingface.co/bert-large-cased/resolve/main/tokenizer.json from cache at /home/ubuntu/.cache/huggingface/transformers/75be22d7750034989358861e325977feda47740e1c3f8a4dc1cb73570aad843e.2b9a196704f2f183fe3f4b48d6e662dba8203fdcb3346bfa896831378edf6f97
[INFO|tokenization_utils_base.py:1707] 2021-06-10 14:54:23,500 >> loading file https://huggingface.co/bert-large-cased/resolve/main/added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 14:54:23,501 >> loading file https://huggingface.co/bert-large-cased/resolve/main/special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 14:54:23,501 >> loading file https://huggingface.co/bert-large-cased/resolve/main/tokenizer_config.json from cache at /home/ubuntu/.cache/huggingface/transformers/45d2aa048795efc7b12791662c188d5e3aa2f9ac54b2cf3f6e4d7bc6544e3d13.ec5c189f89475aac7d8cbd243960a0655cfadc3d0474da8ff2ed0bf1699c2a5f
[INFO|file_utils.py:1426] 2021-06-10 14:54:23,864 >> https://huggingface.co/bert-large-cased/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpwikp623g

[INFO|file_utils.py:1430] 2021-06-10 14:54:52,012 >> storing https://huggingface.co/bert-large-cased/resolve/main/pytorch_model.bin in cache at /home/ubuntu/.cache/huggingface/transformers/cdd3fa79a58abc10a3331427b8b3d7a13ed15ea2dc5bf6dd67065b007d81f2fb.0749e4f07a7ad43190d183545a30a4899a63bd709586bcc3b30b0f09b025ab3a
[INFO|file_utils.py:1433] 2021-06-10 14:54:52,012 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/cdd3fa79a58abc10a3331427b8b3d7a13ed15ea2dc5bf6dd67065b007d81f2fb.0749e4f07a7ad43190d183545a30a4899a63bd709586bcc3b30b0f09b025ab3a
[INFO|modeling_utils.py:1052] 2021-06-10 14:54:52,013 >> loading weights file https://huggingface.co/bert-large-cased/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/cdd3fa79a58abc10a3331427b8b3d7a13ed15ea2dc5bf6dd67065b007d81f2fb.0749e4f07a7ad43190d183545a30a4899a63bd709586bcc3b30b0f09b025ab3a
[WARNING|modeling_utils.py:1160] 2021-06-10 14:55:05,782 >> Some weights of the model checkpoint at bert-large-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1171] 2021-06-10 14:55:05,782 >> Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-large-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


06/10/2021 14:55:08 - INFO - __main__ -   Sample 409 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'idx': 409, 'input_ids': [101, 1169, 1128, 1855, 1103, 1171, 4015, 1113, 170, 1714, 4932, 102, 4299, 4932, 118, 118, 4299, 11784, 1169, 5156, 1129, 2046, 1120, 170, 1344, 6556, 1118, 1363, 2139, 119, 1130, 1103, 5803, 117, 1211, 2139, 1294, 3102, 118, 118, 2908, 110, 1104, 1147, 4021, 119, 1109, 2074, 112, 188, 1436, 14896, 1116, 113, 1216, 1112, 3036, 10544, 117, 5512, 5631, 117, 4150, 4522, 117, 5180, 21056, 4722, 117, 3620, 20422, 117, 19343, 3902, 117, 4101, 12786, 6922, 117, 1105, 19862, 1986, 8976, 2293, 114, 1169, 1294, 4986, 3078, 110, 1104, 1147, 4021, 1166, 170, 1265, 117, 1229, 14140, 1193, 2869, 14896, 1116, 113, 174, 119, 176, 119, 14868, 4115, 117, 3177, 1592, 3276, 1874, 4421, 117, 160, 14080, 15506, 117, 12971, 20978, 117, 1262, 4889, 139, 102], 'label': 1, 'passage': "Free throw -- Free throws can normally be shot at a high percentage by good players. In the NBA, most players make 70--80% of their attempts. The league's best shooters (such as Steve Nash, Rick Barry, Ray Allen, José Calderón, Stephen Curry, Reggie Miller, Kevin Durant, and Dirk Nowitzki) can make roughly 90% of their attempts over a season, while notoriously poor shooters (e.g. Dwight Howard, DeAndre Jordan, Wilt Chamberlain, Andre Drummond, Andris Biedrins, Chris Dudley, Ben Wallace, Shaquille O'Neal, and Dennis Rodman) may struggle to make 50% of them. During a foul shot, a player's foot must be completely behind the foul line. If a player lines up with part of his or her foot on the line, a violation is called and the shot does not count. Foul shots are worth one point.", 'question': 'can you hit the backboard on a free throw', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}.
06/10/2021 14:55:08 - INFO - __main__ -   Sample 4506 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 4506, 'input_ids': [101, 1169, 178, 3952, 170, 2998, 1107, 1103, 6346, 1443, 170, 1862, 4134, 102, 11121, 4134, 118, 118, 1109, 1862, 4134, 1110, 1136, 2320, 1113, 14889, 6346, 119, 1438, 117, 2960, 1104, 170, 1862, 4134, 17356, 1103, 14889, 1555, 1121, 1217, 1682, 1106, 1862, 1103, 8926, 1191, 1122, 17617, 5576, 21091, 4121, 1895, 132, 1216, 1112, 1121, 3290, 117, 2112, 2553, 1496, 117, 1137, 22475, 7680, 119, 5723, 6346, 1336, 4303, 1561, 2044, 2998, 6346, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 1, 'passage': 'Return address -- The return address is not required on postal mail. However, lack of a return address prevents the postal service from being able to return the item if it proves undeliverable; such as from damage, postage due, or invalid destination. Such mail may otherwise become dead letter mail.', 'question': 'can i send a letter in the mail without a return address', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.

[INFO|trainer.py:491] 2021-06-10 14:55:12,747 >> The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: passage, idx, question.
[INFO|trainer.py:491] 2021-06-10 14:55:12,748 >> The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: passage, idx, question.
[INFO|trainer.py:1013] 2021-06-10 14:55:12,909 >> ***** Running training *****
[INFO|trainer.py:1014] 2021-06-10 14:55:12,909 >>   Num examples = 9427
[INFO|trainer.py:1015] 2021-06-10 14:55:12,909 >>   Num Epochs = 40
[INFO|trainer.py:1016] 2021-06-10 14:55:12,910 >>   Instantaneous batch size per device = 16
[INFO|trainer.py:1017] 2021-06-10 14:55:12,910 >>   Total train batch size (w. parallel, distributed & accumulation) = 128
[INFO|trainer.py:1018] 2021-06-10 14:55:12,910 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:1019] 2021-06-10 14:55:12,910 >>   Total optimization steps = 2920

[INFO|configuration_utils.py:329] 2021-06-10 15:39:26,880 >> Configuration saved in ../data/output/boolq/bert-large-cased/checkpoint-500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 15:39:31,038 >> Model weights saved in ../data/output/boolq/bert-large-cased/checkpoint-500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 15:39:31,039 >> tokenizer config file saved in ../data/output/boolq/bert-large-cased/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 15:39:31,040 >> Special tokens file saved in ../data/output/boolq/bert-large-cased/checkpoint-500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 16:23:46,502 >> Configuration saved in ../data/output/boolq/bert-large-cased/checkpoint-1000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 16:23:50,741 >> Model weights saved in ../data/output/boolq/bert-large-cased/checkpoint-1000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 16:23:50,742 >> tokenizer config file saved in ../data/output/boolq/bert-large-cased/checkpoint-1000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 16:23:50,742 >> Special tokens file saved in ../data/output/boolq/bert-large-cased/checkpoint-1000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 17:08:07,220 >> Configuration saved in ../data/output/boolq/bert-large-cased/checkpoint-1500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 17:08:11,446 >> Model weights saved in ../data/output/boolq/bert-large-cased/checkpoint-1500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 17:08:11,447 >> tokenizer config file saved in ../data/output/boolq/bert-large-cased/checkpoint-1500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 17:08:11,448 >> Special tokens file saved in ../data/output/boolq/bert-large-cased/checkpoint-1500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 17:52:14,193 >> Configuration saved in ../data/output/boolq/bert-large-cased/checkpoint-2000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 17:52:18,450 >> Model weights saved in ../data/output/boolq/bert-large-cased/checkpoint-2000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 17:52:18,451 >> tokenizer config file saved in ../data/output/boolq/bert-large-cased/checkpoint-2000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 17:52:18,452 >> Special tokens file saved in ../data/output/boolq/bert-large-cased/checkpoint-2000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 18:36:17,325 >> Configuration saved in ../data/output/boolq/bert-large-cased/checkpoint-2500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 18:36:21,666 >> Model weights saved in ../data/output/boolq/bert-large-cased/checkpoint-2500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 18:36:21,667 >> tokenizer config file saved in ../data/output/boolq/bert-large-cased/checkpoint-2500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 18:36:21,667 >> Special tokens file saved in ../data/output/boolq/bert-large-cased/checkpoint-2500/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)



[INFO|trainer.py:1648] 2021-06-10 19:13:20,659 >> Saving model checkpoint to ../data/output/boolq/bert-large-cased
[INFO|configuration_utils.py:329] 2021-06-10 19:13:20,660 >> Configuration saved in ../data/output/boolq/bert-large-cased/config.json
[INFO|modeling_utils.py:831] 2021-06-10 19:13:24,949 >> Model weights saved in ../data/output/boolq/bert-large-cased/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 19:13:24,950 >> tokenizer config file saved in ../data/output/boolq/bert-large-cased/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 19:13:24,950 >> Special tokens file saved in ../data/output/boolq/bert-large-cased/special_tokens_map.json
[INFO|trainer_pt_utils.py:722] 2021-06-10 19:13:24,988 >> ***** train metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,988 >>   epoch                      =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   init_mem_cpu_alloc_delta   =     1220MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   init_mem_cpu_peaked_delta  =     1265MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   init_mem_gpu_alloc_delta   =     1273MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   init_mem_gpu_peaked_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   train_mem_cpu_alloc_delta  =       60MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   train_mem_cpu_peaked_delta =      355MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   train_mem_gpu_alloc_delta  =     3819MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   train_mem_gpu_peaked_delta =     4480MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   train_runtime              = 4:18:07.55
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   train_samples              =       9427
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:13:24,989 >>   train_samples_per_second   =      0.189
{'loss': 0.287, 'learning_rate': 3.315068493150685e-05, 'epoch': 6.84}
{'loss': 0.0405, 'learning_rate': 2.6301369863013698e-05, 'epoch': 13.69}
{'loss': 0.0128, 'learning_rate': 1.945205479452055e-05, 'epoch': 20.54}
{'loss': 0.0052, 'learning_rate': 1.2602739726027398e-05, 'epoch': 27.39}
{'loss': 0.001, 'learning_rate': 5.753424657534246e-06, 'epoch': 34.24}
{'train_runtime': 15487.5514, 'train_samples_per_second': 0.189, 'epoch': 39.99}
06/10/2021 19:13:24 - INFO - __main__ -   *** Evaluate ***
[INFO|trainer.py:491] 2021-06-10 19:13:25,040 >> The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: passage, idx, question.
[INFO|trainer.py:1865] 2021-06-10 19:13:25,041 >> ***** Running Evaluation *****
[INFO|trainer.py:1866] 2021-06-10 19:13:25,041 >>   Num examples = 3270
[INFO|trainer.py:1867] 2021-06-10 19:13:25,041 >>   Batch size = 8

[INFO|trainer_pt_utils.py:722] 2021-06-10 19:14:16,368 >> ***** eval metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   epoch                     =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_accuracy             =      0.737
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_loss                 =     2.5658
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_mem_cpu_alloc_delta  =       10MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_mem_cpu_peaked_delta =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_mem_gpu_alloc_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_mem_gpu_peaked_delta =       44MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_runtime              = 0:00:51.27
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_samples              =       3270
[INFO|trainer_pt_utils.py:727] 2021-06-10 19:14:16,369 >>   eval_samples_per_second   =     63.776
+ (( i++ ))
+ (( i<7 ))
+ bash run_boolq.sh albert-base-v2
+ export WANDB_DISABLED=True
+ WANDB_DISABLED=True
+ model_name=albert-base-v2
+ out_task=boolq
+ out_dir=../data/output/boolq/albert-base-v2
+ max_seq_len=128
+ batch_size=16
+ lr=4e-5
+ epochs=40
+ accum_steps=8
+ train_path=../data/boolq/train.json
+ val_path=../data/boolq/val.json
+ python ../glue_bench/run_glue.py --model_name_or_path albert-base-v2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 4e-5 --num_train_epochs 40 --output_dir ../data/output/boolq/albert-base-v2 --train_file ../data/boolq/train.json --validation_file ../data/boolq/val.json --gradient_accumulation_steps 8 --overwrite_output_dir
Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
06/10/2021 19:14:20 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
06/10/2021 19:14:20 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir=../data/output/boolq/albert-base-v2, overwrite_output_dir=True, do_train=True, do_eval=True, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=8, gradient_accumulation_steps=8, eval_accumulation_steps=None, learning_rate=4e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=40.0, max_steps=-1, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=0, logging_dir=runs/Jun10_19-14-20_ip-172-31-14-159, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=500, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name=../data/output/boolq/albert-base-v2, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, length_column_name=length, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, _n_gpu=1, mp_parameters=)
06/10/2021 19:14:20 - INFO - __main__ -   load a local file for train: ../data/boolq/train.json
06/10/2021 19:14:20 - INFO - __main__ -   load a local file for validation: ../data/boolq/val.json
Using custom data configuration default
Reusing dataset json (/home/ubuntu/.cache/huggingface/datasets/json/default-87e14794c70b068f/0.0.0/fb88b12bd94767cb0cc7eedcd82ea1f402d2162addc03a37e81d4f8dc7313ad9)
[INFO|file_utils.py:1426] 2021-06-10 19:14:21,748 >> https://huggingface.co/albert-base-v2/resolve/main/config.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpvprsabjf

[INFO|file_utils.py:1430] 2021-06-10 19:14:22,079 >> storing https://huggingface.co/albert-base-v2/resolve/main/config.json in cache at /home/ubuntu/.cache/huggingface/transformers/e48be00f755a5f765e36a32885e8d6a573081df3321c9e19428d12abadf7dba2.b8f28145885741cf994c0e8a97b724f6c974460c297002145e48e511d2496e88
[INFO|file_utils.py:1433] 2021-06-10 19:14:22,079 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/e48be00f755a5f765e36a32885e8d6a573081df3321c9e19428d12abadf7dba2.b8f28145885741cf994c0e8a97b724f6c974460c297002145e48e511d2496e88
[INFO|configuration_utils.py:491] 2021-06-10 19:14:22,079 >> loading configuration file https://huggingface.co/albert-base-v2/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/e48be00f755a5f765e36a32885e8d6a573081df3321c9e19428d12abadf7dba2.b8f28145885741cf994c0e8a97b724f6c974460c297002145e48e511d2496e88
[INFO|configuration_utils.py:527] 2021-06-10 19:14:22,080 >> Model config AlbertConfig {
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu_new",
  "hidden_dropout_prob": 0,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "vocab_size": 30000
}

[INFO|configuration_utils.py:491] 2021-06-10 19:14:22,413 >> loading configuration file https://huggingface.co/albert-base-v2/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/e48be00f755a5f765e36a32885e8d6a573081df3321c9e19428d12abadf7dba2.b8f28145885741cf994c0e8a97b724f6c974460c297002145e48e511d2496e88
[INFO|configuration_utils.py:527] 2021-06-10 19:14:22,414 >> Model config AlbertConfig {
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu_new",
  "hidden_dropout_prob": 0,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "vocab_size": 30000
}

[INFO|file_utils.py:1426] 2021-06-10 19:14:22,741 >> https://huggingface.co/albert-base-v2/resolve/main/spiece.model not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmp5x6sxdoa

[INFO|file_utils.py:1430] 2021-06-10 19:14:23,305 >> storing https://huggingface.co/albert-base-v2/resolve/main/spiece.model in cache at /home/ubuntu/.cache/huggingface/transformers/10be6ce6d3508f1fdce98a57a574283b47c055228c1235f8686f039287ff8174.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|file_utils.py:1433] 2021-06-10 19:14:23,306 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/10be6ce6d3508f1fdce98a57a574283b47c055228c1235f8686f039287ff8174.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|file_utils.py:1426] 2021-06-10 19:14:23,641 >> https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpvmfxa036

[INFO|file_utils.py:1430] 2021-06-10 19:14:24,295 >> storing https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json in cache at /home/ubuntu/.cache/huggingface/transformers/828a43aa4b9d07e2b7d3be7c6bc10a3ae6e16e8d9c3a0c557783639de9eaeb1b.670e237d152dd53ef77575d4f4a6cd34158db03128fe4f63437ce0d5992bac74
[INFO|file_utils.py:1433] 2021-06-10 19:14:24,295 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/828a43aa4b9d07e2b7d3be7c6bc10a3ae6e16e8d9c3a0c557783639de9eaeb1b.670e237d152dd53ef77575d4f4a6cd34158db03128fe4f63437ce0d5992bac74
[INFO|tokenization_utils_base.py:1707] 2021-06-10 19:14:25,298 >> loading file https://huggingface.co/albert-base-v2/resolve/main/spiece.model from cache at /home/ubuntu/.cache/huggingface/transformers/10be6ce6d3508f1fdce98a57a574283b47c055228c1235f8686f039287ff8174.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|tokenization_utils_base.py:1707] 2021-06-10 19:14:25,298 >> loading file https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json from cache at /home/ubuntu/.cache/huggingface/transformers/828a43aa4b9d07e2b7d3be7c6bc10a3ae6e16e8d9c3a0c557783639de9eaeb1b.670e237d152dd53ef77575d4f4a6cd34158db03128fe4f63437ce0d5992bac74
[INFO|tokenization_utils_base.py:1707] 2021-06-10 19:14:25,298 >> loading file https://huggingface.co/albert-base-v2/resolve/main/added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 19:14:25,298 >> loading file https://huggingface.co/albert-base-v2/resolve/main/special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 19:14:25,298 >> loading file https://huggingface.co/albert-base-v2/resolve/main/tokenizer_config.json from cache at None
[INFO|file_utils.py:1426] 2021-06-10 19:14:25,688 >> https://huggingface.co/albert-base-v2/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpoe3y3wqr

[INFO|file_utils.py:1430] 2021-06-10 19:14:27,099 >> storing https://huggingface.co/albert-base-v2/resolve/main/pytorch_model.bin in cache at /home/ubuntu/.cache/huggingface/transformers/bf1986d976e9a8320cbd3a0597e610bf299d639ce31b7ca581cbf54be3aaa6d3.d6d54047dfe6ae844e3bf6e7a7d0aff71cb598d3df019361e076ba7639b1da9b
[INFO|file_utils.py:1433] 2021-06-10 19:14:27,100 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/bf1986d976e9a8320cbd3a0597e610bf299d639ce31b7ca581cbf54be3aaa6d3.d6d54047dfe6ae844e3bf6e7a7d0aff71cb598d3df019361e076ba7639b1da9b
[INFO|modeling_utils.py:1052] 2021-06-10 19:14:27,100 >> loading weights file https://huggingface.co/albert-base-v2/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/bf1986d976e9a8320cbd3a0597e610bf299d639ce31b7ca581cbf54be3aaa6d3.d6d54047dfe6ae844e3bf6e7a7d0aff71cb598d3df019361e076ba7639b1da9b
[WARNING|modeling_utils.py:1160] 2021-06-10 19:14:27,538 >> Some weights of the model checkpoint at albert-base-v2 were not used when initializing AlbertForSequenceClassification: ['predictions.bias', 'predictions.LayerNorm.weight', 'predictions.LayerNorm.bias', 'predictions.dense.weight', 'predictions.dense.bias', 'predictions.decoder.weight', 'predictions.decoder.bias']
- This IS expected if you are initializing AlbertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing AlbertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1171] 2021-06-10 19:14:27,538 >> Some weights of AlbertForSequenceClassification were not initialized from the model checkpoint at albert-base-v2 and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


06/10/2021 19:14:31 - INFO - __main__ -   Sample 409 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'idx': 409, 'input_ids': [2, 92, 42, 770, 14, 97, 2806, 27, 21, 551, 3814, 3, 551, 3814, 13, 8, 8, 551, 13566, 92, 4147, 44, 999, 35, 21, 183, 5780, 34, 254, 1007, 9, 19, 14, 5811, 15, 127, 1007, 233, 3201, 8, 8, 24516, 16, 66, 3265, 9, 14, 278, 22, 18, 246, 12674, 18, 13, 5, 4289, 28, 2228, 9710, 15, 4163, 5150, 15, 2375, 3675, 15, 2712, 27168, 15, 2526, 18239, 15, 20885, 3022, 15, 3480, 13188, 38, 15, 17, 19209, 130, 7565, 1520, 6, 92, 233, 4457, 16370, 16, 66, 3265, 84, 21, 198, 15, 133, 11904, 102, 1696, 12674, 18, 13, 5, 62, 9, 263, 9, 16183, 3444, 15, 121, 18832, 3759, 15, 5266, 38, 16083, 15, 17, 99, 20547, 15, 17, 2777, 1732, 69, 3], 'label': 1, 'passage': "Free throw -- Free throws can normally be shot at a high percentage by good players. In the NBA, most players make 70--80% of their attempts. The league's best shooters (such as Steve Nash, Rick Barry, Ray Allen, José Calderón, Stephen Curry, Reggie Miller, Kevin Durant, and Dirk Nowitzki) can make roughly 90% of their attempts over a season, while notoriously poor shooters (e.g. Dwight Howard, DeAndre Jordan, Wilt Chamberlain, Andre Drummond, Andris Biedrins, Chris Dudley, Ben Wallace, Shaquille O'Neal, and Dennis Rodman) may struggle to make 50% of them. During a foul shot, a player's foot must be completely behind the foul line. If a player lines up with part of his or her foot on the line, a violation is called and the shot does not count. Foul shots are worth one point.", 'question': 'can you hit the backboard on a free throw', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}.
06/10/2021 19:14:31 - INFO - __main__ -   Sample 4506 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 4506, 'input_ids': [2, 92, 31, 2660, 21, 1748, 19, 14, 4216, 366, 21, 788, 3218, 3, 788, 3218, 13, 8, 8, 14, 788, 3218, 25, 52, 1390, 27, 11680, 4216, 9, 207, 15, 1792, 16, 21, 788, 3218, 2501, 18, 14, 11680, 365, 37, 142, 777, 20, 788, 14, 9101, 100, 32, 4220, 18, 367, 29731, 579, 73, 145, 28, 37, 2308, 15, 25040, 397, 15, 54, 16671, 6970, 9, 145, 4216, 123, 3190, 533, 828, 1748, 4216, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 1, 'passage': 'Return address -- The return address is not required on postal mail. However, lack of a return address prevents the postal service from being able to return the item if it proves undeliverable; such as from damage, postage due, or invalid destination. Such mail may otherwise become dead letter mail.', 'question': 'can i send a letter in the mail without a return address', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.

[INFO|trainer.py:491] 2021-06-10 19:14:35,153 >> The following columns in the training set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: idx, question, passage.
[INFO|trainer.py:491] 2021-06-10 19:14:35,153 >> The following columns in the evaluation set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: idx, question, passage.
[INFO|trainer.py:1013] 2021-06-10 19:14:35,245 >> ***** Running training *****
[INFO|trainer.py:1014] 2021-06-10 19:14:35,245 >>   Num examples = 9427
[INFO|trainer.py:1015] 2021-06-10 19:14:35,245 >>   Num Epochs = 40
[INFO|trainer.py:1016] 2021-06-10 19:14:35,245 >>   Instantaneous batch size per device = 16
[INFO|trainer.py:1017] 2021-06-10 19:14:35,245 >>   Total train batch size (w. parallel, distributed & accumulation) = 128
[INFO|trainer.py:1018] 2021-06-10 19:14:35,246 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:1019] 2021-06-10 19:14:35,246 >>   Total optimization steps = 2920

[INFO|configuration_utils.py:329] 2021-06-10 19:30:05,988 >> Configuration saved in ../data/output/boolq/albert-base-v2/checkpoint-500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 19:30:06,124 >> Model weights saved in ../data/output/boolq/albert-base-v2/checkpoint-500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 19:30:06,124 >> tokenizer config file saved in ../data/output/boolq/albert-base-v2/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 19:30:06,125 >> Special tokens file saved in ../data/output/boolq/albert-base-v2/checkpoint-500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 19:45:43,140 >> Configuration saved in ../data/output/boolq/albert-base-v2/checkpoint-1000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 19:45:43,224 >> Model weights saved in ../data/output/boolq/albert-base-v2/checkpoint-1000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 19:45:43,228 >> tokenizer config file saved in ../data/output/boolq/albert-base-v2/checkpoint-1000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 19:45:43,228 >> Special tokens file saved in ../data/output/boolq/albert-base-v2/checkpoint-1000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 20:01:19,518 >> Configuration saved in ../data/output/boolq/albert-base-v2/checkpoint-1500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 20:01:19,621 >> Model weights saved in ../data/output/boolq/albert-base-v2/checkpoint-1500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 20:01:19,621 >> tokenizer config file saved in ../data/output/boolq/albert-base-v2/checkpoint-1500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 20:01:19,622 >> Special tokens file saved in ../data/output/boolq/albert-base-v2/checkpoint-1500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 20:16:51,780 >> Configuration saved in ../data/output/boolq/albert-base-v2/checkpoint-2000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 20:16:51,865 >> Model weights saved in ../data/output/boolq/albert-base-v2/checkpoint-2000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 20:16:51,866 >> tokenizer config file saved in ../data/output/boolq/albert-base-v2/checkpoint-2000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 20:16:51,866 >> Special tokens file saved in ../data/output/boolq/albert-base-v2/checkpoint-2000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 20:32:27,606 >> Configuration saved in ../data/output/boolq/albert-base-v2/checkpoint-2500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 20:32:27,691 >> Model weights saved in ../data/output/boolq/albert-base-v2/checkpoint-2500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 20:32:27,691 >> tokenizer config file saved in ../data/output/boolq/albert-base-v2/checkpoint-2500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 20:32:27,692 >> Special tokens file saved in ../data/output/boolq/albert-base-v2/checkpoint-2500/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)



[INFO|trainer.py:1648] 2021-06-10 20:45:35,684 >> Saving model checkpoint to ../data/output/boolq/albert-base-v2
[INFO|configuration_utils.py:329] 2021-06-10 20:45:35,685 >> Configuration saved in ../data/output/boolq/albert-base-v2/config.json
[INFO|modeling_utils.py:831] 2021-06-10 20:45:35,766 >> Model weights saved in ../data/output/boolq/albert-base-v2/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 20:45:35,767 >> tokenizer config file saved in ../data/output/boolq/albert-base-v2/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 20:45:35,767 >> Special tokens file saved in ../data/output/boolq/albert-base-v2/special_tokens_map.json
[INFO|trainer_pt_utils.py:722] 2021-06-10 20:45:35,803 >> ***** train metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   epoch                      =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   init_mem_cpu_alloc_delta   =     2436MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   init_mem_cpu_peaked_delta  =       37MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   init_mem_gpu_alloc_delta   =       44MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   init_mem_gpu_peaked_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   train_mem_cpu_alloc_delta  =      112MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   train_mem_cpu_peaked_delta =        7MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   train_mem_gpu_alloc_delta  =      139MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   train_mem_gpu_peaked_delta =     2363MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   train_runtime              = 1:31:00.29
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   train_samples              =       9427
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:45:35,804 >>   train_samples_per_second   =      0.535
{'loss': 0.3732, 'learning_rate': 3.315068493150685e-05, 'epoch': 6.84}
{'loss': 0.0872, 'learning_rate': 2.6301369863013698e-05, 'epoch': 13.69}
{'loss': 0.0397, 'learning_rate': 1.945205479452055e-05, 'epoch': 20.54}
{'loss': 0.0229, 'learning_rate': 1.2602739726027398e-05, 'epoch': 27.39}
{'loss': 0.0031, 'learning_rate': 5.753424657534246e-06, 'epoch': 34.24}
{'train_runtime': 5460.2919, 'train_samples_per_second': 0.535, 'epoch': 39.99}
06/10/2021 20:45:35 - INFO - __main__ -   *** Evaluate ***
[INFO|trainer.py:491] 2021-06-10 20:45:35,847 >> The following columns in the evaluation set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: idx, question, passage.
[INFO|trainer.py:1865] 2021-06-10 20:45:35,848 >> ***** Running Evaluation *****
[INFO|trainer.py:1866] 2021-06-10 20:45:35,848 >>   Num examples = 3270
[INFO|trainer.py:1867] 2021-06-10 20:45:35,848 >>   Batch size = 8

[INFO|trainer_pt_utils.py:722] 2021-06-10 20:46:03,368 >> ***** eval metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,368 >>   epoch                     =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_accuracy             =     0.7532
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_loss                 =       1.74
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_mem_cpu_alloc_delta  =        7MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_mem_cpu_peaked_delta =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_mem_gpu_alloc_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_mem_gpu_peaked_delta =       54MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_runtime              = 0:00:27.47
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_samples              =       3270
[INFO|trainer_pt_utils.py:727] 2021-06-10 20:46:03,369 >>   eval_samples_per_second   =    119.036
+ (( i++ ))
+ (( i<7 ))
+ bash run_boolq.sh albert-large-v2
+ export WANDB_DISABLED=True
+ WANDB_DISABLED=True
+ model_name=albert-large-v2
+ out_task=boolq
+ out_dir=../data/output/boolq/albert-large-v2
+ max_seq_len=128
+ batch_size=16
+ lr=4e-5
+ epochs=40
+ accum_steps=8
+ train_path=../data/boolq/train.json
+ val_path=../data/boolq/val.json
+ python ../glue_bench/run_glue.py --model_name_or_path albert-large-v2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 4e-5 --num_train_epochs 40 --output_dir ../data/output/boolq/albert-large-v2 --train_file ../data/boolq/train.json --validation_file ../data/boolq/val.json --gradient_accumulation_steps 8 --overwrite_output_dir
Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
06/10/2021 20:46:06 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
06/10/2021 20:46:06 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir=../data/output/boolq/albert-large-v2, overwrite_output_dir=True, do_train=True, do_eval=True, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=8, gradient_accumulation_steps=8, eval_accumulation_steps=None, learning_rate=4e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=40.0, max_steps=-1, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=0, logging_dir=runs/Jun10_20-46-06_ip-172-31-14-159, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=500, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name=../data/output/boolq/albert-large-v2, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, length_column_name=length, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, _n_gpu=1, mp_parameters=)
06/10/2021 20:46:06 - INFO - __main__ -   load a local file for train: ../data/boolq/train.json
06/10/2021 20:46:06 - INFO - __main__ -   load a local file for validation: ../data/boolq/val.json
Using custom data configuration default
Reusing dataset json (/home/ubuntu/.cache/huggingface/datasets/json/default-87e14794c70b068f/0.0.0/fb88b12bd94767cb0cc7eedcd82ea1f402d2162addc03a37e81d4f8dc7313ad9)
[INFO|file_utils.py:1426] 2021-06-10 20:46:07,410 >> https://huggingface.co/albert-large-v2/resolve/main/config.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpxjwnffov

[INFO|file_utils.py:1430] 2021-06-10 20:46:07,739 >> storing https://huggingface.co/albert-large-v2/resolve/main/config.json in cache at /home/ubuntu/.cache/huggingface/transformers/b2da41a68a8020e0d5923bb74adfb48c33df97683e143ca33ad6e52a3e05d70d.06fa0ad0d486db01b65880587686cbf167e4c2d52e242d574fac416eda16c32d
[INFO|file_utils.py:1433] 2021-06-10 20:46:07,739 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/b2da41a68a8020e0d5923bb74adfb48c33df97683e143ca33ad6e52a3e05d70d.06fa0ad0d486db01b65880587686cbf167e4c2d52e242d574fac416eda16c32d
[INFO|configuration_utils.py:491] 2021-06-10 20:46:07,740 >> loading configuration file https://huggingface.co/albert-large-v2/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/b2da41a68a8020e0d5923bb74adfb48c33df97683e143ca33ad6e52a3e05d70d.06fa0ad0d486db01b65880587686cbf167e4c2d52e242d574fac416eda16c32d
[INFO|configuration_utils.py:527] 2021-06-10 20:46:07,740 >> Model config AlbertConfig {
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu_new",
  "hidden_dropout_prob": 0,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 16,
  "num_hidden_groups": 1,
  "num_hidden_layers": 24,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "vocab_size": 30000
}

[INFO|configuration_utils.py:491] 2021-06-10 20:46:08,071 >> loading configuration file https://huggingface.co/albert-large-v2/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/b2da41a68a8020e0d5923bb74adfb48c33df97683e143ca33ad6e52a3e05d70d.06fa0ad0d486db01b65880587686cbf167e4c2d52e242d574fac416eda16c32d
[INFO|configuration_utils.py:527] 2021-06-10 20:46:08,071 >> Model config AlbertConfig {
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu_new",
  "hidden_dropout_prob": 0,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 16,
  "num_hidden_groups": 1,
  "num_hidden_layers": 24,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "vocab_size": 30000
}

[INFO|file_utils.py:1426] 2021-06-10 20:46:08,411 >> https://huggingface.co/albert-large-v2/resolve/main/spiece.model not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpglob5xm0

[INFO|file_utils.py:1430] 2021-06-10 20:46:08,980 >> storing https://huggingface.co/albert-large-v2/resolve/main/spiece.model in cache at /home/ubuntu/.cache/huggingface/transformers/b4bd5194827ca5bc0342e0421aace72462c676f37679a440862cf3ee46f95f48.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|file_utils.py:1433] 2021-06-10 20:46:08,981 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/b4bd5194827ca5bc0342e0421aace72462c676f37679a440862cf3ee46f95f48.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|file_utils.py:1426] 2021-06-10 20:46:09,307 >> https://huggingface.co/albert-large-v2/resolve/main/tokenizer.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpljvv8qig

[INFO|file_utils.py:1430] 2021-06-10 20:46:09,945 >> storing https://huggingface.co/albert-large-v2/resolve/main/tokenizer.json in cache at /home/ubuntu/.cache/huggingface/transformers/8f1144987c0a5fcedc8808300dc830a4a00787ceaccb85e9f913ef047103bd89.670e237d152dd53ef77575d4f4a6cd34158db03128fe4f63437ce0d5992bac74
[INFO|file_utils.py:1433] 2021-06-10 20:46:09,946 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/8f1144987c0a5fcedc8808300dc830a4a00787ceaccb85e9f913ef047103bd89.670e237d152dd53ef77575d4f4a6cd34158db03128fe4f63437ce0d5992bac74
[INFO|tokenization_utils_base.py:1707] 2021-06-10 20:46:10,931 >> loading file https://huggingface.co/albert-large-v2/resolve/main/spiece.model from cache at /home/ubuntu/.cache/huggingface/transformers/b4bd5194827ca5bc0342e0421aace72462c676f37679a440862cf3ee46f95f48.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|tokenization_utils_base.py:1707] 2021-06-10 20:46:10,931 >> loading file https://huggingface.co/albert-large-v2/resolve/main/tokenizer.json from cache at /home/ubuntu/.cache/huggingface/transformers/8f1144987c0a5fcedc8808300dc830a4a00787ceaccb85e9f913ef047103bd89.670e237d152dd53ef77575d4f4a6cd34158db03128fe4f63437ce0d5992bac74
[INFO|tokenization_utils_base.py:1707] 2021-06-10 20:46:10,931 >> loading file https://huggingface.co/albert-large-v2/resolve/main/added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 20:46:10,931 >> loading file https://huggingface.co/albert-large-v2/resolve/main/special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 20:46:10,931 >> loading file https://huggingface.co/albert-large-v2/resolve/main/tokenizer_config.json from cache at None
[INFO|file_utils.py:1426] 2021-06-10 20:46:11,324 >> https://huggingface.co/albert-large-v2/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmp0tmcbvfc

[INFO|file_utils.py:1430] 2021-06-10 20:46:14,291 >> storing https://huggingface.co/albert-large-v2/resolve/main/pytorch_model.bin in cache at /home/ubuntu/.cache/huggingface/transformers/4552fda677d63af6d9acdc968e0a4bfb09ef6994bbb37c065ea6533cf8dc0977.4ffe1a3c3f6feb9b16e8d8811a495b5ca957bb71b1215ae0960c2b06f2e7d9bd
[INFO|file_utils.py:1433] 2021-06-10 20:46:14,291 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/4552fda677d63af6d9acdc968e0a4bfb09ef6994bbb37c065ea6533cf8dc0977.4ffe1a3c3f6feb9b16e8d8811a495b5ca957bb71b1215ae0960c2b06f2e7d9bd
[INFO|modeling_utils.py:1052] 2021-06-10 20:46:14,291 >> loading weights file https://huggingface.co/albert-large-v2/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/4552fda677d63af6d9acdc968e0a4bfb09ef6994bbb37c065ea6533cf8dc0977.4ffe1a3c3f6feb9b16e8d8811a495b5ca957bb71b1215ae0960c2b06f2e7d9bd
[WARNING|modeling_utils.py:1160] 2021-06-10 20:46:14,933 >> Some weights of the model checkpoint at albert-large-v2 were not used when initializing AlbertForSequenceClassification: ['predictions.bias', 'predictions.LayerNorm.weight', 'predictions.LayerNorm.bias', 'predictions.dense.weight', 'predictions.dense.bias', 'predictions.decoder.weight', 'predictions.decoder.bias']
- This IS expected if you are initializing AlbertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing AlbertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1171] 2021-06-10 20:46:14,933 >> Some weights of AlbertForSequenceClassification were not initialized from the model checkpoint at albert-large-v2 and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


06/10/2021 20:46:18 - INFO - __main__ -   Sample 409 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'idx': 409, 'input_ids': [2, 92, 42, 770, 14, 97, 2806, 27, 21, 551, 3814, 3, 551, 3814, 13, 8, 8, 551, 13566, 92, 4147, 44, 999, 35, 21, 183, 5780, 34, 254, 1007, 9, 19, 14, 5811, 15, 127, 1007, 233, 3201, 8, 8, 24516, 16, 66, 3265, 9, 14, 278, 22, 18, 246, 12674, 18, 13, 5, 4289, 28, 2228, 9710, 15, 4163, 5150, 15, 2375, 3675, 15, 2712, 27168, 15, 2526, 18239, 15, 20885, 3022, 15, 3480, 13188, 38, 15, 17, 19209, 130, 7565, 1520, 6, 92, 233, 4457, 16370, 16, 66, 3265, 84, 21, 198, 15, 133, 11904, 102, 1696, 12674, 18, 13, 5, 62, 9, 263, 9, 16183, 3444, 15, 121, 18832, 3759, 15, 5266, 38, 16083, 15, 17, 99, 20547, 15, 17, 2777, 1732, 69, 3], 'label': 1, 'passage': "Free throw -- Free throws can normally be shot at a high percentage by good players. In the NBA, most players make 70--80% of their attempts. The league's best shooters (such as Steve Nash, Rick Barry, Ray Allen, José Calderón, Stephen Curry, Reggie Miller, Kevin Durant, and Dirk Nowitzki) can make roughly 90% of their attempts over a season, while notoriously poor shooters (e.g. Dwight Howard, DeAndre Jordan, Wilt Chamberlain, Andre Drummond, Andris Biedrins, Chris Dudley, Ben Wallace, Shaquille O'Neal, and Dennis Rodman) may struggle to make 50% of them. During a foul shot, a player's foot must be completely behind the foul line. If a player lines up with part of his or her foot on the line, a violation is called and the shot does not count. Foul shots are worth one point.", 'question': 'can you hit the backboard on a free throw', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}.
06/10/2021 20:46:18 - INFO - __main__ -   Sample 4506 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 4506, 'input_ids': [2, 92, 31, 2660, 21, 1748, 19, 14, 4216, 366, 21, 788, 3218, 3, 788, 3218, 13, 8, 8, 14, 788, 3218, 25, 52, 1390, 27, 11680, 4216, 9, 207, 15, 1792, 16, 21, 788, 3218, 2501, 18, 14, 11680, 365, 37, 142, 777, 20, 788, 14, 9101, 100, 32, 4220, 18, 367, 29731, 579, 73, 145, 28, 37, 2308, 15, 25040, 397, 15, 54, 16671, 6970, 9, 145, 4216, 123, 3190, 533, 828, 1748, 4216, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 1, 'passage': 'Return address -- The return address is not required on postal mail. However, lack of a return address prevents the postal service from being able to return the item if it proves undeliverable; such as from damage, postage due, or invalid destination. Such mail may otherwise become dead letter mail.', 'question': 'can i send a letter in the mail without a return address', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.

[INFO|trainer.py:491] 2021-06-10 20:46:22,552 >> The following columns in the training set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: idx, question, passage.
[INFO|trainer.py:491] 2021-06-10 20:46:22,553 >> The following columns in the evaluation set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: idx, question, passage.
[INFO|trainer.py:1013] 2021-06-10 20:46:22,633 >> ***** Running training *****
[INFO|trainer.py:1014] 2021-06-10 20:46:22,634 >>   Num examples = 9427
[INFO|trainer.py:1015] 2021-06-10 20:46:22,634 >>   Num Epochs = 40
[INFO|trainer.py:1016] 2021-06-10 20:46:22,634 >>   Instantaneous batch size per device = 16
[INFO|trainer.py:1017] 2021-06-10 20:46:22,634 >>   Total train batch size (w. parallel, distributed & accumulation) = 128
[INFO|trainer.py:1018] 2021-06-10 20:46:22,634 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:1019] 2021-06-10 20:46:22,634 >>   Total optimization steps = 2920

[INFO|configuration_utils.py:329] 2021-06-10 21:16:37,985 >> Configuration saved in ../data/output/boolq/albert-large-v2/checkpoint-500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 21:16:38,210 >> Model weights saved in ../data/output/boolq/albert-large-v2/checkpoint-500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 21:16:38,211 >> tokenizer config file saved in ../data/output/boolq/albert-large-v2/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 21:16:38,211 >> Special tokens file saved in ../data/output/boolq/albert-large-v2/checkpoint-500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 21:42:54,016 >> Configuration saved in ../data/output/boolq/albert-large-v2/checkpoint-1000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 21:42:54,233 >> Model weights saved in ../data/output/boolq/albert-large-v2/checkpoint-1000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 21:42:54,234 >> tokenizer config file saved in ../data/output/boolq/albert-large-v2/checkpoint-1000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 21:42:54,234 >> Special tokens file saved in ../data/output/boolq/albert-large-v2/checkpoint-1000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 22:09:12,604 >> Configuration saved in ../data/output/boolq/albert-large-v2/checkpoint-1500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 22:09:12,820 >> Model weights saved in ../data/output/boolq/albert-large-v2/checkpoint-1500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 22:09:12,820 >> tokenizer config file saved in ../data/output/boolq/albert-large-v2/checkpoint-1500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 22:09:12,821 >> Special tokens file saved in ../data/output/boolq/albert-large-v2/checkpoint-1500/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 22:35:26,040 >> Configuration saved in ../data/output/boolq/albert-large-v2/checkpoint-2000/config.json
[INFO|modeling_utils.py:831] 2021-06-10 22:35:26,259 >> Model weights saved in ../data/output/boolq/albert-large-v2/checkpoint-2000/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 22:35:26,259 >> tokenizer config file saved in ../data/output/boolq/albert-large-v2/checkpoint-2000/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 22:35:26,260 >> Special tokens file saved in ../data/output/boolq/albert-large-v2/checkpoint-2000/special_tokens_map.json

[INFO|configuration_utils.py:329] 2021-06-10 23:01:38,931 >> Configuration saved in ../data/output/boolq/albert-large-v2/checkpoint-2500/config.json
[INFO|modeling_utils.py:831] 2021-06-10 23:01:39,148 >> Model weights saved in ../data/output/boolq/albert-large-v2/checkpoint-2500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 23:01:39,149 >> tokenizer config file saved in ../data/output/boolq/albert-large-v2/checkpoint-2500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 23:01:39,149 >> Special tokens file saved in ../data/output/boolq/albert-large-v2/checkpoint-2500/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)



[INFO|trainer.py:1648] 2021-06-10 23:23:39,429 >> Saving model checkpoint to ../data/output/boolq/albert-large-v2
[INFO|configuration_utils.py:329] 2021-06-10 23:23:39,430 >> Configuration saved in ../data/output/boolq/albert-large-v2/config.json
[INFO|modeling_utils.py:831] 2021-06-10 23:23:39,636 >> Model weights saved in ../data/output/boolq/albert-large-v2/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-10 23:23:39,637 >> tokenizer config file saved in ../data/output/boolq/albert-large-v2/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-10 23:23:39,637 >> Special tokens file saved in ../data/output/boolq/albert-large-v2/special_tokens_map.json
[INFO|trainer_pt_utils.py:722] 2021-06-10 23:23:39,674 >> ***** train metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,674 >>   epoch                      =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   init_mem_cpu_alloc_delta   =     2435MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   init_mem_cpu_peaked_delta  =       58MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   init_mem_gpu_alloc_delta   =       67MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   init_mem_gpu_peaked_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   train_mem_cpu_alloc_delta  =       62MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   train_mem_cpu_peaked_delta =       56MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   train_mem_gpu_alloc_delta  =      202MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   train_mem_gpu_peaked_delta =     6226MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   train_runtime              = 2:37:16.52
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   train_samples              =       9427
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:23:39,675 >>   train_samples_per_second   =      0.309
{'loss': 0.6734, 'learning_rate': 3.315068493150685e-05, 'epoch': 6.84}
{'loss': 0.6716, 'learning_rate': 2.6301369863013698e-05, 'epoch': 13.69}
{'loss': 0.6701, 'learning_rate': 1.945205479452055e-05, 'epoch': 20.54}
{'loss': 0.6701, 'learning_rate': 1.2602739726027398e-05, 'epoch': 27.39}
{'loss': 0.6704, 'learning_rate': 5.753424657534246e-06, 'epoch': 34.24}
{'train_runtime': 9436.5254, 'train_samples_per_second': 0.309, 'epoch': 39.99}
06/10/2021 23:23:39 - INFO - __main__ -   *** Evaluate ***
[INFO|trainer.py:491] 2021-06-10 23:23:39,719 >> The following columns in the evaluation set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: idx, question, passage.
[INFO|trainer.py:1865] 2021-06-10 23:23:39,720 >> ***** Running Evaluation *****
[INFO|trainer.py:1866] 2021-06-10 23:23:39,720 >>   Num examples = 3270
[INFO|trainer.py:1867] 2021-06-10 23:23:39,720 >>   Batch size = 8

[INFO|trainer_pt_utils.py:722] 2021-06-10 23:24:19,131 >> ***** eval metrics *****
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   epoch                     =      39.99
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_accuracy             =     0.6217
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_loss                 =     0.6632
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_mem_cpu_alloc_delta  =        7MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_mem_cpu_peaked_delta =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_mem_gpu_alloc_delta  =        0MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_mem_gpu_peaked_delta =       72MB
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_runtime              = 0:00:39.35
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_samples              =       3270
[INFO|trainer_pt_utils.py:727] 2021-06-10 23:24:19,131 >>   eval_samples_per_second   =      83.08
+ (( i++ ))
+ (( i<7 ))
+ bash run_boolq.sh albert-xlarge-v2
+ export WANDB_DISABLED=True
+ WANDB_DISABLED=True
+ model_name=albert-xlarge-v2
+ out_task=boolq
+ out_dir=../data/output/boolq/albert-xlarge-v2
+ max_seq_len=128
+ batch_size=16
+ lr=4e-5
+ epochs=40
+ accum_steps=8
+ train_path=../data/boolq/train.json
+ val_path=../data/boolq/val.json
+ python ../glue_bench/run_glue.py --model_name_or_path albert-xlarge-v2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 4e-5 --num_train_epochs 40 --output_dir ../data/output/boolq/albert-xlarge-v2 --train_file ../data/boolq/train.json --validation_file ../data/boolq/val.json --gradient_accumulation_steps 8 --overwrite_output_dir
Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
06/10/2021 23:24:21 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
06/10/2021 23:24:21 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir=../data/output/boolq/albert-xlarge-v2, overwrite_output_dir=True, do_train=True, do_eval=True, do_predict=False, evaluation_strategy=IntervalStrategy.NO, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=8, gradient_accumulation_steps=8, eval_accumulation_steps=None, learning_rate=4e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=40.0, max_steps=-1, lr_scheduler_type=SchedulerType.LINEAR, warmup_ratio=0.0, warmup_steps=0, logging_dir=runs/Jun10_23-24-21_ip-172-31-14-159, logging_strategy=IntervalStrategy.STEPS, logging_first_step=False, logging_steps=500, save_strategy=IntervalStrategy.STEPS, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=O1, fp16_backend=auto, fp16_full_eval=False, local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name=../data/output/boolq/albert-xlarge-v2, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, sharded_ddp=[], deepspeed=None, label_smoothing_factor=0.0, adafactor=False, group_by_length=False, length_column_name=length, report_to=['tensorboard'], ddp_find_unused_parameters=None, dataloader_pin_memory=True, skip_memory_metrics=False, _n_gpu=1, mp_parameters=)
06/10/2021 23:24:21 - INFO - __main__ -   load a local file for train: ../data/boolq/train.json
06/10/2021 23:24:21 - INFO - __main__ -   load a local file for validation: ../data/boolq/val.json
Using custom data configuration default
Reusing dataset json (/home/ubuntu/.cache/huggingface/datasets/json/default-87e14794c70b068f/0.0.0/fb88b12bd94767cb0cc7eedcd82ea1f402d2162addc03a37e81d4f8dc7313ad9)
[INFO|file_utils.py:1426] 2021-06-10 23:24:22,921 >> https://huggingface.co/albert-xlarge-v2/resolve/main/config.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpxdo1qj3i

[INFO|file_utils.py:1430] 2021-06-10 23:24:23,256 >> storing https://huggingface.co/albert-xlarge-v2/resolve/main/config.json in cache at /home/ubuntu/.cache/huggingface/transformers/90f3e255eb2906b428c0235b8d52302cb5d11f6073952f0dd2b13bd96da5c04e.13a56b68bb98b0a796545b9d47760fe5993a05e8461b1a0a6821d278f575f6db
[INFO|file_utils.py:1433] 2021-06-10 23:24:23,256 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/90f3e255eb2906b428c0235b8d52302cb5d11f6073952f0dd2b13bd96da5c04e.13a56b68bb98b0a796545b9d47760fe5993a05e8461b1a0a6821d278f575f6db
[INFO|configuration_utils.py:491] 2021-06-10 23:24:23,256 >> loading configuration file https://huggingface.co/albert-xlarge-v2/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/90f3e255eb2906b428c0235b8d52302cb5d11f6073952f0dd2b13bd96da5c04e.13a56b68bb98b0a796545b9d47760fe5993a05e8461b1a0a6821d278f575f6db
[INFO|configuration_utils.py:527] 2021-06-10 23:24:23,257 >> Model config AlbertConfig {
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu_new",
  "hidden_dropout_prob": 0,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 16,
  "num_hidden_groups": 1,
  "num_hidden_layers": 24,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "vocab_size": 30000
}

[INFO|configuration_utils.py:491] 2021-06-10 23:24:23,585 >> loading configuration file https://huggingface.co/albert-xlarge-v2/resolve/main/config.json from cache at /home/ubuntu/.cache/huggingface/transformers/90f3e255eb2906b428c0235b8d52302cb5d11f6073952f0dd2b13bd96da5c04e.13a56b68bb98b0a796545b9d47760fe5993a05e8461b1a0a6821d278f575f6db
[INFO|configuration_utils.py:527] 2021-06-10 23:24:23,585 >> Model config AlbertConfig {
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu_new",
  "hidden_dropout_prob": 0,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 16,
  "num_hidden_groups": 1,
  "num_hidden_layers": 24,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.5.1",
  "type_vocab_size": 2,
  "vocab_size": 30000
}

[INFO|file_utils.py:1426] 2021-06-10 23:24:23,912 >> https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmptnl733rj

[INFO|file_utils.py:1430] 2021-06-10 23:24:24,480 >> storing https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model in cache at /home/ubuntu/.cache/huggingface/transformers/806115ace5c54db49637a1c887c48171c57c22add4ff802b469447fd70c6e67c.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|file_utils.py:1433] 2021-06-10 23:24:24,480 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/806115ace5c54db49637a1c887c48171c57c22add4ff802b469447fd70c6e67c.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|file_utils.py:1426] 2021-06-10 23:24:24,807 >> https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer.json not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmppt1am8vu

[INFO|file_utils.py:1430] 2021-06-10 23:24:25,459 >> storing https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer.json in cache at /home/ubuntu/.cache/huggingface/transformers/546025946742cdd9258e9b056df8c6fdfab4cdaa39e80495f4a0bcfb12a42733.a3d46b3322815ee0cf8b043ac6ce8f8ff7dcb0e910f7536529d6caf0dff51411
[INFO|file_utils.py:1433] 2021-06-10 23:24:25,459 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/546025946742cdd9258e9b056df8c6fdfab4cdaa39e80495f4a0bcfb12a42733.a3d46b3322815ee0cf8b043ac6ce8f8ff7dcb0e910f7536529d6caf0dff51411
[INFO|tokenization_utils_base.py:1707] 2021-06-10 23:24:26,443 >> loading file https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model from cache at /home/ubuntu/.cache/huggingface/transformers/806115ace5c54db49637a1c887c48171c57c22add4ff802b469447fd70c6e67c.d6110e25022b713452eb83d5bfa8ae64530995a93d8e694fe52e05aa85dd3a7d
[INFO|tokenization_utils_base.py:1707] 2021-06-10 23:24:26,444 >> loading file https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer.json from cache at /home/ubuntu/.cache/huggingface/transformers/546025946742cdd9258e9b056df8c6fdfab4cdaa39e80495f4a0bcfb12a42733.a3d46b3322815ee0cf8b043ac6ce8f8ff7dcb0e910f7536529d6caf0dff51411
[INFO|tokenization_utils_base.py:1707] 2021-06-10 23:24:26,444 >> loading file https://huggingface.co/albert-xlarge-v2/resolve/main/added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 23:24:26,444 >> loading file https://huggingface.co/albert-xlarge-v2/resolve/main/special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:1707] 2021-06-10 23:24:26,444 >> loading file https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer_config.json from cache at None
[INFO|file_utils.py:1426] 2021-06-10 23:24:26,828 >> https://huggingface.co/albert-xlarge-v2/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /home/ubuntu/.cache/huggingface/transformers/tmpb2gucsrb

[INFO|file_utils.py:1430] 2021-06-10 23:24:31,830 >> storing https://huggingface.co/albert-xlarge-v2/resolve/main/pytorch_model.bin in cache at /home/ubuntu/.cache/huggingface/transformers/6f88777c22cc772881d46aeaa5339534f7f71722a8b1e67b96bc0228c03ca292.083421ab5997af191e6dad6145de3145ccd45f977dbcddf7cdbfa4302c33e997
[INFO|file_utils.py:1433] 2021-06-10 23:24:31,830 >> creating metadata file for /home/ubuntu/.cache/huggingface/transformers/6f88777c22cc772881d46aeaa5339534f7f71722a8b1e67b96bc0228c03ca292.083421ab5997af191e6dad6145de3145ccd45f977dbcddf7cdbfa4302c33e997
[INFO|modeling_utils.py:1052] 2021-06-10 23:24:31,830 >> loading weights file https://huggingface.co/albert-xlarge-v2/resolve/main/pytorch_model.bin from cache at /home/ubuntu/.cache/huggingface/transformers/6f88777c22cc772881d46aeaa5339534f7f71722a8b1e67b96bc0228c03ca292.083421ab5997af191e6dad6145de3145ccd45f977dbcddf7cdbfa4302c33e997
[WARNING|modeling_utils.py:1160] 2021-06-10 23:24:33,902 >> Some weights of the model checkpoint at albert-xlarge-v2 were not used when initializing AlbertForSequenceClassification: ['predictions.bias', 'predictions.LayerNorm.weight', 'predictions.LayerNorm.bias', 'predictions.dense.weight', 'predictions.dense.bias', 'predictions.decoder.weight', 'predictions.decoder.bias']
- This IS expected if you are initializing AlbertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing AlbertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1171] 2021-06-10 23:24:33,902 >> Some weights of AlbertForSequenceClassification were not initialized from the model checkpoint at albert-xlarge-v2 and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


06/10/2021 23:24:36 - INFO - __main__ -   Sample 409 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'idx': 409, 'input_ids': [2, 92, 42, 770, 14, 97, 2806, 27, 21, 551, 3814, 3, 551, 3814, 13, 8, 8, 551, 13566, 92, 4147, 44, 999, 35, 21, 183, 5780, 34, 254, 1007, 9, 19, 14, 5811, 15, 127, 1007, 233, 3201, 8, 8, 24516, 16, 66, 3265, 9, 14, 278, 22, 18, 246, 12674, 18, 13, 5, 4289, 28, 2228, 9710, 15, 4163, 5150, 15, 2375, 3675, 15, 2712, 27168, 15, 2526, 18239, 15, 20885, 3022, 15, 3480, 13188, 38, 15, 17, 19209, 130, 7565, 1520, 6, 92, 233, 4457, 16370, 16, 66, 3265, 84, 21, 198, 15, 133, 11904, 102, 1696, 12674, 18, 13, 5, 62, 9, 263, 9, 16183, 3444, 15, 121, 18832, 3759, 15, 5266, 38, 16083, 15, 17, 99, 20547, 15, 17, 2777, 1732, 69, 3], 'label': 1, 'passage': "Free throw -- Free throws can normally be shot at a high percentage by good players. In the NBA, most players make 70--80% of their attempts. The league's best shooters (such as Steve Nash, Rick Barry, Ray Allen, José Calderón, Stephen Curry, Reggie Miller, Kevin Durant, and Dirk Nowitzki) can make roughly 90% of their attempts over a season, while notoriously poor shooters (e.g. Dwight Howard, DeAndre Jordan, Wilt Chamberlain, Andre Drummond, Andris Biedrins, Chris Dudley, Ben Wallace, Shaquille O'Neal, and Dennis Rodman) may struggle to make 50% of them. During a foul shot, a player's foot must be completely behind the foul line. If a player lines up with part of his or her foot on the line, a violation is called and the shot does not count. Foul shots are worth one point.", 'question': 'can you hit the backboard on a free throw', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}.
06/10/2021 23:24:36 - INFO - __main__ -   Sample 4506 of the training set: {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'idx': 4506, 'input_ids': [2, 92, 31, 2660, 21, 1748, 19, 14, 4216, 366, 21, 788, 3218, 3, 788, 3218, 13, 8, 8, 14, 788, 3218, 25, 52, 1390, 27, 11680, 4216, 9, 207, 15, 1792, 16, 21, 788, 3218, 2501, 18, 14, 11680, 365, 37, 142, 777, 20, 788, 14, 9101, 100, 32, 4220, 18, 367, 29731, 579, 73, 145, 28, 37, 2308, 15, 25040, 397, 15, 54, 16671, 6970, 9, 145, 4216, 123, 3190, 533, 828, 1748, 4216, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label': 1, 'passage': 'Return address -- The return address is not required on postal mail. However, lack of a return address prevents the postal service from being able to return the item if it proves undeliverable; such as from damage, postage due, or invalid destination. Such mail may otherwise become dead letter mail.', 'question': 'can i send a letter in the mail without a return address', 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.

[INFO|trainer.py:491] 2021-06-10 23:24:40,395 >> The following columns in the training set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: passage, question, idx.
[INFO|trainer.py:491] 2021-06-10 23:24:40,395 >> The following columns in the evaluation set  don't have a corresponding argument in `AlbertForSequenceClassification.forward` and have been ignored: passage, question, idx.
[INFO|trainer.py:1013] 2021-06-10 23:24:40,482 >> ***** Running training *****
[INFO|trainer.py:1014] 2021-06-10 23:24:40,482 >>   Num examples = 9427
[INFO|trainer.py:1015] 2021-06-10 23:24:40,482 >>   Num Epochs = 40
[INFO|trainer.py:1016] 2021-06-10 23:24:40,482 >>   Instantaneous batch size per device = 16
[INFO|trainer.py:1017] 2021-06-10 23:24:40,482 >>   Total train batch size (w. parallel, distributed & accumulation) = 128
[INFO|trainer.py:1018] 2021-06-10 23:24:40,482 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:1019] 2021-06-10 23:24:40,482 >>   Total optimization steps = 2920

[INFO|configuration_utils.py:329] 2021-06-11 00:50:16,179 >> Configuration saved in ../data/output/boolq/albert-xlarge-v2/checkpoint-500/config.json
[INFO|modeling_utils.py:831] 2021-06-11 00:50:17,065 >> Model weights saved in ../data/output/boolq/albert-xlarge-v2/checkpoint-500/pytorch_model.bin
[INFO|tokenization_utils_base.py:1901] 2021-06-11 00:50:17,066 >> tokenizer config file saved in ../data/output/boolq/albert-xlarge-v2/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:1907] 2021-06-11 00:50:17,066 >> Special tokens file saved in ../data/output/boolq/albert-xlarge-v2/checkpoint-500/special_tokens_map.json
