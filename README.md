## Set up instructions

```
# Create necessary directories
mkdir data data_bin models checkpoint

# Download the DM Math dataset
gsutil cp gs://mathematics-dataset/mathematics_dataset-v1.0.tar.gz ./data
tar xvf data/mathematics_dataset-v1.0.tar.gz -C data

# Download the tokenizer
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json -P data
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe -P data

# Download the 125M model
wget https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_125m.tar.gz -P models
tar xvf models/en_dense_lm_125m.tar.gz -C models

```

## Run instructions:

- Both `fairseq-preprocess` and `fairseq-train` are being passed `--cpu`. Delete this if you're running on GPU
- If you've run it at all unsuccessfully before, make sure to delete files it produces as it checks for their existence before creating them.

### For training the model on a cluster:

- This link may be key: https://github.com/pytorch/fairseq/tree/moe#training-moe-language-models
- The link was found in this readme: https://github.com/pytorch/fairseq/tree/main/examples/moe_lm#mixture-of-expert-models

### Where plamb is stuck:

If you just run the code you should get this error:

```shell
Traceback (most recent call last):
  File "/Users/plamb/miniconda3/envs/m1-math/bin/fairseq-train", line 8, in <module>
    sys.exit(cli_main())
  File "/Users/plamb/miniconda3/envs/m1-math/lib/python3.8/site-packages/fairseq_cli/train.py", line 557, in cli_main
    distributed_utils.call_main(cfg, main)
  File "/Users/plamb/miniconda3/envs/m1-math/lib/python3.8/site-packages/fairseq/distributed/utils.py", line 369, in call_main
    main(cfg, **kwargs)
  File "/Users/plamb/miniconda3/envs/m1-math/lib/python3.8/site-packages/fairseq_cli/train.py", line 164, in main
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
  File "/Users/plamb/miniconda3/envs/m1-math/lib/python3.8/site-packages/fairseq/checkpoint_utils.py", line 248, in load_checkpoint
    extra_state = trainer.load_checkpoint(
  File "/Users/plamb/miniconda3/envs/m1-math/lib/python3.8/site-packages/fairseq/trainer.py", line 628, in load_checkpoint
    and itr_state["iterations_in_epoch"] == 0
KeyError: 'iterations_in_epoch'
```
I opened the file and just set `itr_state["iterations_in_epoch"] = 0`

And training actually kicked off!

So we need to figure out why this value isn't getting set



### I believe i fixed the below problem (so this can be ignored):
- I am currently getting the below errors when running `fairseq-train`
- ```
     RuntimeError: Error(s) in loading state_dict for TransformerLanguageModel:
     size mismatch for decoder.embed_tokens.weight: copying a param with shape torch.Size([51200, 1024]) from checkpoint, the shape in current model is torch.Size([1200, 1024]).
     size mismatch for decoder.output_projection.weight: copying a param with shape torch.Size([51200, 1024]) from checkpoint, the shape in current model is torch.Size([1200, 1024]).
  ```
- The reason this is occuring is because of the below comment:
  - https://github.com/pytorch/fairseq/issues/367#issuecomment-439690689
- As such, I re-wrote the code to grab `dict.txt` out of the `/model/name-here/` path and pass it to `preproces_data`
- It then gets passed into `preprocess_data` via `--srcdict` and `--only-source` (note that source stands for source language, you can see more here [here](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess))
- Unfortunately this did not solve the error, you can see it occur as preprocess runs:
  - The logs output: ` INFO | fairseq_cli.preprocess | [None] Dictionary: 51200 types`
  - The number of which matches my above error: `size mismatch copying torch.Size([51200, 1024]) to torch.Size([1200, 1024])`
  - In essence, the fix is somehow using or passing a dictionary that forces the number of 'types' down to 1200
  - I'm not sure what dict.txt even is in this context

### Config for em_dense_lm_125m (printed):

```yaml
common:
  _name: null
  no_progress_bar: false
  log_interval: 25
  log_format: json
  wandb_project: null
  seed: 1
  cpu: false
  tpu: false
  bf16: false
  memory_efficient_bf16: false
  fp16: true
  memory_efficient_fp16: false
  fp16_no_flatten_grads: true
  fp16_init_scale: 128
  fp16_scale_window: null
  fp16_scale_tolerance: 0.0
  min_loss_scale: 0.0001
  threshold_loss_scale: null
  user_dir: null
  empty_cache_freq: 0
  all_gather_list_size: 16384
  model_parallel_size: 1
  quantization_config_path: null
  profile: false
common_eval:
  _name: null
  path: null
  post_process: null
  quiet: false
  model_overrides: '{}'
  results_path: null
distributed_training:
  _name: null
  distributed_world_size: 64
  distributed_rank: 0
  distributed_backend: nccl
  device_id: 0
  local_rank: 0
  distributed_no_spawn: false
  ddp_backend: c10d
  bucket_cap_mb: 25
  fix_batches_to_gpus: false
  find_unused_parameters: false
  fast_stat_sync: false
  broadcast_buffers: false
  distributed_wrapper: DDP
  slowmo_momentum: null
  slowmo_algorithm: LocalSGD
  localsgd_frequency: 3
  nprocs_per_node: 8
  pipeline_model_parallel: false
  pipeline_balance: null
  pipeline_devices: null
  pipeline_chunks: 0
  pipeline_encoder_balance: null
  pipeline_encoder_devices: null
  pipeline_decoder_balance: null
  pipeline_decoder_devices: null
  pipeline_checkpoint: never
  zero_sharding: os
  tpu: false
  distributed_num_procs: 8
dataset:
  _name: null
  num_workers: 2
  skip_invalid_size_inputs_valid_test: false
  max_tokens: null
  batch_size: 1
  required_batch_size_multiple: 1
  required_seq_len_multiple: 1
  dataset_impl: null
  data_buffer_size: 10
  train_subset: train
  valid_subset: valid
  validate_interval: 1
  validate_interval_updates: 0
  validate_after_updates: 0
  fixed_validation_seed: null
  disable_validation: false
  max_tokens_valid: null
  batch_size_valid: 1
  curriculum: 0
  gen_subset: test
  num_shards: 1
  shard_id: 0
optimization:
  _name: null
  max_epoch: 0
  max_update: 572204
  stop_time_hours: 0.0
  clip_norm: 0.0
  sentence_avg: false
  update_freq:
  - 4
  lr:
  - 0.0003
  min_lr: -1.0
  use_bmuf: false
checkpoint:
  _name: null
  finetune_from_model: null
  reset_dataloader: false
  reset_lr_scheduler: false
  reset_meters: false
  reset_optimizer: false
  optimizer_overrides: '{}'
  save_interval: 1
  save_interval_updates: 10000
  keep_interval_updates: -1
  keep_last_epochs: -1
  keep_best_checkpoints: -1
  no_save: false
  no_epoch_checkpoints: true
  no_last_checkpoints: false
  no_save_optimizer_state: false
  best_checkpoint_metric: loss
  maximize_best_checkpoint_metric: false
  patience: -1
  checkpoint_suffix: ''
  checkpoint_shard_count: 1
  model_parallel_size: 1
  distributed_rank: 0
bmuf:
  _name: null
  block_lr: 1.0
  block_momentum: 0.875
  global_sync_iter: 50
  warmup_iterations: 500
  use_nbm: false
  average_sync: false
  distributed_world_size: 64
generation:
  _name: null
  beam: 5
  nbest: 1
  max_len_a: 0.0
  max_len_b: 200
  min_len: 1
  match_source_len: false
  unnormalized: false
  no_early_stop: false
  no_beamable_mm: false
  lenpen: 1.0
  unkpen: 0.0
  replace_unk: null
  sacrebleu: false
  score_reference: false
  prefix_size: 0
  no_repeat_ngram_size: 0
  sampling: false
  sampling_topk: -1
  sampling_topp: -1.0
  constraints: null
  temperature: 1.0
  diverse_beam_groups: -1
  diverse_beam_strength: 0.5
  diversity_rate: -1.0
  print_alignment: false
  print_step: false
  lm_path: null
  lm_weight: 0.0
  iter_decode_eos_penalty: 0.0
  iter_decode_max_iter: 10
  iter_decode_force_max_iter: false
  iter_decode_with_beam: 1
  iter_decode_with_external_reranker: false
  retain_iter_history: false
  retain_dropout: false
  retain_dropout_modules: null
  decoding_format: null
  no_seed_provided: false
eval_lm:
  _name: null
  output_word_probs: false
  output_word_stats: false
  context_window: 0
  softmax_batch: 9223372036854775807
interactive:
  _name: null
  buffer_size: 0
  input: '-'
model:
  _name: transformer_lm_gpt2_small
  activation_fn: gelu
  dropout: 0.0
  attention_dropout: 0.0
  activation_dropout: 0.0
  relu_dropout: 0.0
  decoder_embed_dim: 1024
  decoder_output_dim: 1024
  decoder_input_dim: 1024
  decoder_ffn_embed_dim: 4096
  decoder_layers: 24
  decoder_attention_heads: 16
  decoder_normalize_before: true
  no_decoder_final_norm: false
  adaptive_softmax_cutoff: null
  adaptive_softmax_dropout: 0.0
  adaptive_softmax_factor: 4.0
  no_token_positional_embeddings: false
  share_decoder_input_output_embed: true
  character_embeddings: false
  character_filters: '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256),
    (7, 256)]'
  character_embedding_dim: 4
  char_embedder_highway_layers: 2
  adaptive_input: false
  adaptive_input_factor: 4.0
  adaptive_input_cutoff: null
  tie_adaptive_weights: false
  tie_adaptive_proj: false
  decoder_learned_pos: false
  decoder_layerdrop: 0.0
  decoder_layers_to_keep: null
  layernorm_embedding: false
  no_scale_embedding: false
  checkpoint_activations: false
  quant_noise_pq: 0.0
  quant_noise_pq_block_size: 8
  quant_noise_scalar: 0.0
  add_bos_token: false
  tokens_per_sample: 2048
  max_target_positions: 2048
  tpu: false
task:
  _name: language_modeling
  data: .
  sample_break_mode: none
  tokens_per_sample: 2048
  output_dictionary_size: -1
  self_target: false
  future_target: false
  past_target: false
  add_bos_token: false
  max_source_positions: null
  max_target_positions: null
  shorten_method: none
  shorten_data_split_list: ''
  seed: 1
  dataset_impl: null
  data_buffer_size: 10
  tpu: false
criterion:
  _name: cross_entropy
  sentence_avg: false
optimizer:
  _name: adam
  adam_betas: (0.9, 0.98)
  adam_eps: 1.0e-08
  weight_decay: 0.01
  use_old_adam: false
  tpu: false
  lr:
  - 0.0003
lr_scheduler:
  no_progress_bar: false
  log_interval: 25
  log_format: json
  wandb_project: null
  seed: 1
  cpu: false
  tpu: false
  bf16: false
  memory_efficient_bf16: false
  fp16: true
  memory_efficient_fp16: false
  fp16_no_flatten_grads: true
  fp16_init_scale: 128
  fp16_scale_window: null
  fp16_scale_tolerance: 0.0
  min_loss_scale: 0.0001
  threshold_loss_scale: null
  user_dir: null
  empty_cache_freq: 0
  all_gather_list_size: 16384
  model_parallel_size: 1
  quantization_config_path: null
  profile: false
  criterion: cross_entropy
  tokenizer: null
  bpe: null
  optimizer: adam
  lr_scheduler: polynomial_decay
  scoring: bleu
  task: language_modeling
  num_workers: 2
  skip_invalid_size_inputs_valid_test: false
  max_tokens: null
  batch_size: 1
  required_batch_size_multiple: 1
  required_seq_len_multiple: 1
  dataset_impl: null
  data_buffer_size: 10
  train_subset: train
  valid_subset: valid
  validate_interval: 1
  validate_interval_updates: 0
  validate_after_updates: 0
  fixed_validation_seed: null
  disable_validation: false
  max_tokens_valid: null
  batch_size_valid: 1
  curriculum: 0
  gen_subset: test
  num_shards: 1
  shard_id: 0
  distributed_world_size: 64
  distributed_rank: 0
  distributed_backend: nccl
  device_id: 0
  local_rank: 0
  distributed_no_spawn: false
  ddp_backend: c10d
  bucket_cap_mb: 25
  fix_batches_to_gpus: false
  find_unused_parameters: false
  fast_stat_sync: false
  broadcast_buffers: false
  distributed_wrapper: DDP
  slowmo_momentum: null
  slowmo_algorithm: LocalSGD
  localsgd_frequency: 3
  nprocs_per_node: 8
  pipeline_model_parallel: false
  pipeline_balance: null
  pipeline_devices: null
  pipeline_chunks: 0
  pipeline_encoder_balance: null
  pipeline_encoder_devices: null
  pipeline_decoder_balance: null
  pipeline_decoder_devices: null
  pipeline_checkpoint: never
  zero_sharding: os
  arch: transformer_lm_gpt2_small
  max_epoch: 0
  max_update: 572204
  stop_time_hours: 0
  clip_norm: 0.0
  sentence_avg: false
  update_freq:
  - 4
  lr:
  - 0.0003
  min_lr: -1.0
  use_bmuf: false
  finetune_from_model: null
  reset_dataloader: false
  reset_lr_scheduler: false
  reset_meters: false
  reset_optimizer: false
  optimizer_overrides: '{}'
  save_interval: 1
  save_interval_updates: 10000
  keep_interval_updates: -1
  keep_last_epochs: -1
  keep_best_checkpoints: -1
  no_save: false
  no_epoch_checkpoints: true
  no_last_checkpoints: false
  no_save_optimizer_state: false
  best_checkpoint_metric: loss
  maximize_best_checkpoint_metric: false
  patience: -1
  checkpoint_suffix: ''
  checkpoint_shard_count: 1
  sample_break_mode: none
  tokens_per_sample: 2048
  output_dictionary_size: -1
  self_target: false
  future_target: false
  past_target: false
  add_bos_token: false
  max_source_positions: null
  max_target_positions: null
  shorten_method: none
  shorten_data_split_list: ''
  adam_betas: (0.9, 0.98)
  adam_eps: 1.0e-08
  weight_decay: 0.01
  use_old_adam: false
  force_anneal: null
  warmup_updates: 715
  end_learning_rate: 0.0
  power: 1.0
  total_num_update: 572204
  pad: 1
  eos: 2
  unk: 3
  share_decoder_input_output_embed: true
  dropout: 0.0
  attention_dropout: 0.0
  no_seed_provided: false
  decoder_embed_dim: 1024
  decoder_ffn_embed_dim: 4096
  decoder_layers: 24
  decoder_attention_heads: 16
  activation_fn: gelu
  adaptive_softmax_cutoff: null
  adaptive_softmax_dropout: 0
  adaptive_softmax_factor: 4
  decoder_learned_pos: false
  decoder_layerdrop: 0
  decoder_layers_to_keep: null
  quant_noise_pq: 0
  quant_noise_pq_block_size: 8
  quant_noise_scalar: 0
  no_token_positional_embeddings: false
  character_embeddings: false
  decoder_output_dim: 1024
  decoder_input_dim: 1024
  decoder_normalize_before: true
  no_decoder_final_norm: false
  adaptive_input: false
  adaptive_input_factor: 4
  adaptive_input_cutoff: null
  tie_adaptive_weights: false
  tie_adaptive_proj: false
  no_scale_embedding: false
  layernorm_embedding: false
  checkpoint_activations: false
  _name: polynomial_decay
scoring:
  _name: bleu
  pad: 1
  eos: 2
  unk: 3
bpe: null
tokenizer: null
```
