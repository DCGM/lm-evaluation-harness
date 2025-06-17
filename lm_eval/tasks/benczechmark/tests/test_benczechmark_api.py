"""
# Test Objectives
 1. test all tasks, 10 shot
 2. test hf and vllm backend
 3. test with smart truncation on/off
 4. test with chat templates on/off
 --- for cases 2  -  4: Let's try all combinations

 5. test certain case with max_seq_len 1024 & 2048
 6. verify the correct metrics are recorded for every task (TODO: finalize this)
    - these can be found in leaderboard metadata
    - every task that reports auroc should also report accuracy and macro_f1
    - every task that reports accuracy should also report auroc and macro_f1
"""
import sys
import os
import time

import pytest
import torch


os.environ["OPENAI_API_KEY"] = "API_KEY"

# Constants
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LIMIT = 12
FEWSHOT = 20
SUM_LOGPROBS = [
    "benczechmark_histcorpus", "benczechmark_essay", "benczechmark_fiction",
    "benczechmark_capek", "benczechmark_correspondence", "benczechmark_havlicek",
    "benczechmark_speeches", "benczechmark_spoken", "benczechmark_dialect"
]

TASKS = [
    "benczechmark_agree",
    "benczechmark_belebele", "benczechmark_snli",
    "benczechmark_subjectivity", "benczechmark_propaganda_argumentace",
    "benczechmark_propaganda_fabulace", "benczechmark_propaganda_nazor",
    "benczechmark_propaganda_strach", "benczechmark_propaganda_zamereni",
    "benczechmark_propaganda_demonizace", "benczechmark_propaganda_lokace",
    "benczechmark_propaganda_relativizace", "benczechmark_propaganda_vina",
    "benczechmark_propaganda_zanr", "benczechmark_propaganda_emoce",
    "benczechmark_propaganda_nalepkovani", "benczechmark_propaganda_rusko",
    "benczechmark_sentiment_mall", "benczechmark_sentiment_fb",
    "benczechmark_sentiment_csfd", "benczechmark_grammarerrorcorrection",
    "benczechmark_cs_naturalquestions", "benczechmark_cs_sqad32",
    "benczechmark_cs_triviaQA", "benczechmark_csfever_nli",
    "benczechmark_ctkfacts_nli", "benczechmark_cs_ner", "benczechmark_hellaswag",
    "benczechmark_klokan_qa", "benczechmark_cs_court_decisions_ner",
    "benczechmark_umimeto_qa", "benczechmark_cermat_mc",
    "benczechmark_cermat_qa", "benczechmark_history_ir",
    "benczechmark_histcorpus", "benczechmark_essay",
    "benczechmark_fiction", "benczechmark_correspondence",
    "benczechmark_havlicek", "benczechmark_spoken",
    "benczechmark_dialect"
]


# Helper function to generate cli arguments based on configurations
def generate_argv(model_type, task, max_seq_len, truncate_strategy, apply_chat_template,
                  normalize_log_probs):
    # Build chat template argument
    chat_template_args = []
    if apply_chat_template:
        chat_template_args.extend(["--apply_chat_template", "--fewshot_as_multiturn"])

    # Set up the model_args based on the backend
    if model_type == "hf":
        model_args = (f"pretrained={MODEL_NAME},dtype=bfloat16,max_length={max_seq_len},"
                      f"truncation=True,normalize_log_probs={normalize_log_probs},trust_remote_code=True")

    elif model_type == "vllm":
        model_args = (f"pretrained={MODEL_NAME},dtype=bfloat16,gpu_memory_utilization=0.9,"
                      f"max_length={max_seq_len},enforce_eager=True,normalize_log_probs={normalize_log_probs},"
                      f"trust_remote_code=True")

    elif model_type == "openai-completions":
        model_args = (f"base_url=http://athena22.fit.vutbr.cz:8000/v1/completions,model=facebook/opt-125m,truncation=True,"
                      f"tokenizer_backend=huggingface,tokenizer=facebook/opt-125m,max_length={max_seq_len},normalize_log_probs={normalize_log_probs},"
                      f"trust_remote_code=True")

    # Conditionally add truncate_strategy if it's not None
    if truncate_strategy is not None:
        model_args += f",truncate_strategy={truncate_strategy}"

    # Create a unique output path using all relevant parameters
    output_path = (
        f"lm_eval/tasks/benczechmark/tests/test_results/output_{task}_model_{model_type}_fewshot_{FEWSHOT}_maxlen_{max_seq_len}"
        f"_trunc_{truncate_strategy}_logp_{normalize_log_probs}_chat_{apply_chat_template}")

    return [
        "lm_eval.py",
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", task,
        "--output_path", output_path,
        "--log_samples",
        "--verbosity", "DEBUG",
        "--num_fewshot", str(FEWSHOT),
        *chat_template_args,  # Unpack chat template args
        "--limit", str(LIMIT)
    ]


# TODO: vllm backend is not ready yet.  It sometimes throws OOM errors in tests.
# MF: I removed it from the tests for now.
# TODO: try more then 1 model
# TODO: try 1 shot and 10 shot

@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("backend", ["openai-completions"])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("apply_chat_template", [True, False])
@pytest.mark.parametrize("truncate_strategy", ["to_max_length", "leave_description"])
def test_task(task, backend, max_seq_len, apply_chat_template, truncate_strategy):
    from lm_eval.__main__ import cli_evaluate
    normalize_log_probs = "True"

    if task in SUM_LOGPROBS:
        normalize_log_probs = "False"
        # Skip invalid combinations
        if truncate_strategy == "leave_description":
            pytest.skip("Invalid combination: truncate_strategy='leave_description' for tasks in SUM_LOGPROBS")
        if apply_chat_template:
            pytest.skip("Invalid combination: apply_chat_template=True for tasks in SUM_LOGPROBS")

    # Generate command-line arguments
    sys.argv = generate_argv(
        model_type=backend,
        task=task,
        max_seq_len=max_seq_len,
        truncate_strategy=truncate_strategy,
        apply_chat_template=apply_chat_template,
        normalize_log_probs=normalize_log_probs
    )

    try:
        # Call the cli_evaluate function to simulate the CLI execution
        cli_evaluate()
    except RuntimeError as e:
        try:
            if "CUDA out of memory." in str(e):
                torch.cuda.empty_cache()
                # Writing OOM detected error to stderr
                sys.stderr.write("OOM detected, emptying cache...\n")
                time.sleep(3)
        finally:
            raise e
