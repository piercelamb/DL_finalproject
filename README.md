plamb's first attempt at fairseq-preprocess. raw_data_filename is a CSV of Question,Answer arithmetic:

```python
preprocess_data = subprocess.run([
        "fairseq-preprocess",
        "--trainpref=data/"+raw_data_filename,
        "--destdir=data/encoded_dataset.csv"
        "--tokenizer=moses",
        "--bpe=fastbpe"
    ])
```

Output from fairseq:
```shell
2022-04-18 07:30:39 | INFO | fairseq_cli.preprocess | Namespace(align_suffix=None, alignfile=None, all_gather_list_size=16384, bf16=False, bpe='fastbpe', checkpoint_shard_count=1, checkpoint_suffix='', cpu=False, criterion='cross_entropy', dataset_impl='mmap', destdir='data/encoded_dataset.csv--tokenizer=moses', empty_cache_freq=0, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, joined_dictionary=False, log_format=None, log_interval=100, lr_scheduler='fixed', memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_progress_bar=False, nwordssrc=-1, nwordstgt=-1, only_source=False, optimizer=None, padding_factor=8, profile=False, quantization_config_path=None, scoring='bleu', seed=1, source_lang=None, srcdict=None, target_lang=None, task='translation', tensorboard_logdir=None, testpref=None, tgtdict=None, threshold_loss_scale=None, thresholdsrc=0, thresholdtgt=0, tokenizer=None, tpu=False, trainpref='data/raw_dataset.csv', user_dir=None, validpref=None, workers=1)
2022-04-18 07:34:19 | INFO | fairseq_cli.preprocess | [None] Dictionary: 17150032 types
2022-04-18 07:36:17 | INFO | fairseq_cli.preprocess | [None] data/raw_dataset.csv: 17150026 sents, 34300052 tokens, 0.0% replaced by <unk>
2022-04-18 07:36:17 | INFO | fairseq_cli.preprocess | [None] Dictionary: 17150032 types
2022-04-18 07:38:19 | INFO | fairseq_cli.preprocess | [None] data/raw_dataset.csv: 17150026 sents, 34300052 tokens, 0.0% replaced by <unk>
2022-04-18 07:38:19 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to data/encoded_dataset.csv--tokenizer=moses
The exit code was: 0
```

It wrote these files:
```
train.None-None.idx
train.None-None.bin
dict.txt
preprocess.log
```

dict.txt looks like this:

![dict output](https://i.imgur.com/o6NlmAi.png)