# Changelog / Things to be aware of!
* __pre-release v0.5__  
    - bugfix: API models are now able to work with SegmentedStrings.
    - Support of to_max_length truncation strategy for SegmentedStrings.

* __pre-release v0.4__  
    - bugfix: F1 was sometimes corresponding to binary F1, and sometimes to macro F1. F1 was not used on leaderboard, however if you used it, you might want to recompute your tasks.
    - minor fixes, BCM tests added

* __pre-release v0.3__   
If you done the evaluation with older version (v0.2), please reevaluate subjectivity task.
And if you have used the first public version (v0.1), please reevaluate subjectivity, belebele, and snli in order to be comparable with benchmark. Be sure to extract leaderboard results (using [compile_log_files.py](https://github.com/MFajcik/benczechmark-leaderboard/blob/master/leaderboard/compile_log_files.py)) on the new results, not the older results.
* __pre-release v0.2__   
Fixes for belebele, snli, and grammarerrorcorrection tasks. The first critical big in prompt (only answer choices were shown to model; not question and neither context). The latter two tasks were using wrong metrics.
Please reevaluate this tasks before submitting your results to leaderboard.

* If the [leaderboard](https://huggingface.co/spaces/CZLC/BenCzechMark) doesn't show up (or shows something like `Results dataset integrity solving`), it means the model tournament is being recomputed (~ 5 hours). This gets done everytime we fix some crucial bug (so after v0.2, v0.3).


# Introduction to  🇨🇿 BenCzechMark Fork
Welcome to 🇨🇿 BenCzechMark for of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Official readme corresponding to the forked version is [here](README_harness.md). The main differences of this fork include:
* Extra switch for aggregating per-token log-probability with average, instead of sum.
  * Useful for multichoice tasks, when choices vary in length. We have an evidence that some models (such as [BUT-FIT/csmpt7b](https://huggingface.co/BUT-FIT/csmpt7b)) tends to prefer shorter choices in such a case.
  * ❗❗❗ Be sure not to use this with language modeling tasks which require __perplexity__ computation!!!
  
* Smart truncation switch which prevents task description from being truncated.
  * For longer samples in vanilla lm-evaluation-harness, the context gets truncated from left. This can cause troubles, as the context is constructed as follows:
  ```text
  <description><few_shot_examples><current_sample><current_continuation> 
  ```
  * Leftmost truncation can then delete the description altogether; we have an evidence (with [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)) that this can have adverse effect on the results.
  * Smart truncation does the following:
    * It starts with lefmost truncation for `<few_shot_samples>`,
    * When there are no `<few_shot_samples>` left, it does leftmost truncation from `description`, followed by `current_sample`,
    * When prefix (composed of `<description><few_shot_examples><current_sample>`) comes close to 20% of the defined max_input_length, the rightmost truncation from suffix is made further.
  * It also works with  chat_templates! (Experimental support).

# Obtaining Results For Leaderboard Submission
- Follow standard lm-harness parameters. We recommend also using our extra functionality (smart truncation and logp averaging). To see how see **Example Usage** below.
- Be sure to specify `outputh_path` and `log_samples` parameters. Your output paths across all tasks (as matched by glob expression) are then aggregated by [compile_log_files.py](https://huggingface.co/spaces/CZLC/BenCzechMark/blob/main/compile_log_files.py) script (see [leaderboard](https://huggingface.co/spaces/CZLC/BenCzechMark) instructions).
# Example Usage
We ran all experiments on slurm cluster ([Karolina](https://www.it4i.cz/en/infrastructure/karolina), 8x 40GB A100 GPUs per node). For comprehensive survey, check out the job scripts, starting with  [jobs/run_benchmark.sh](jobs/run_benchmark.sh).
We follow standard lm-harness options, together with our custom functionality described in introduction. We didn't used chat_templates.

For executing single 🇨🇿 BenCzechMark task, you can run one of the following:

```bash
# CSMPT7b on 1 GPUs
python -m accelerate.commands.launch \
    --dynamo_backend=inductor \
    -m lm_eval \
    --model hf \
    --model_args pretrained=BUT-FIT/csmpt7b,\
dtype=bfloat16,max_length=2048,\
truncation=True,normalize_log_probs=True,\
trust_remote_code=True,truncate_strategy=leave_description \
    --tasks benczechmark_cs_sqad32 \
    --batch_size 2 \
    --output_path results_hf/eval_csmpt7b_benczechmark_cs_sqad32_chat_none_trunc_leave_description \
    --log_samples \
    --verbosity DEBUG \
    --num_fewshot 3
    
# Mistral Nemo on 8 GPUs, 1 node, using HF backend and pipeline parallelism
python -m lm_eval \
    --model hf \
    --model_args pretrained=mistralai/Mistral-Nemo-Instruct-2407,\
dtype=bfloat16,parallelize=True,max_length=2048,\
truncation=True,normalize_log_probs=True,\
trust_remote_code=True,truncate_strategy=leave_description \
    --tasks benczechmark_sentiment_mall \
    --batch_size 8 \
    --output_path results_hf/eval_mistral_nemo_instruct_benczechmark_sentiment_mall_chat_none_trunc_leave_description \
    --log_samples \
    --verbosity DEBUG \
    --num_fewshot 3
# Mixtral on 8 GPUs, 1 node, using VLLM backend and tensor parallelism
python -m lm_eval \
    --model vllm \
    --model_args pretrained=mistralai/Mixtral-8x7B-Instruct-v0.1,\
tensor_parallel_size=8,dtype=bfloat16,\
gpu_memory_utilization=0.8,max_length=2048,\
normalize_log_probs=True,trust_remote_code=True,\
truncate_strategy=leave_description \
    --tasks benczechmark_czechnews \
    --batch_size auto:4 \
    --output_path results_hf/eval_mixtralM_instruct_benczechmark_czechnews_chat_none_trunc_leave_description \
    --log_samples \
    --verbosity DEBUG \
    --num_fewshot 3
```

# Citation

```bibtex
@misc{fajcik2024benczechmarkczechcentricmultitask,
      title={BenCzechMark : A Czech-centric Multitask and Multimetric Benchmark for Large Language Models with Duel Scoring Mechanism}, 
      author={Martin Fajcik and Martin Docekal and Jan Dolezal and Karel Ondrej and Karel Beneš and Jan Kapsa and Pavel Smrz and Alexander Polok and Michal Hradis and Zuzana Neverilova and Ales Horak and Radoslav Sabol and Michal Stefanik and Adam Jirkovsky and David Adamczyk and Petr Hyner and Jan Hula and Hynek Kydlicek},
      year={2024},
      eprint={2412.17933},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.17933}, 
}
```
    
# See jobs/scripts/models/eval_L_vllm_master.sh for multinode evaluation with VLLM.
```
Notes & Tips:
* Notice the usage of our extra switches, such as `truncate_strategy` (options are None / "leave_description"), and `normalize_log_probs` (True triggers averaging). 
* `batch_size: auto` sometimes causes CUDA OOM errors. We usually ran all tasks with auto, and those which failed were rerun with fixed batch size.
