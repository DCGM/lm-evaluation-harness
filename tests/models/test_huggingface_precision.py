"""Likelihood precision regressions without model or dataset downloads."""

import unittest
from types import SimpleNamespace

import torch
import transformers

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM


class FixedLogitsHFLM(HFLM):
    def __init__(self, dtype, normalize_log_probs):
        LM.__init__(self)
        self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
        self._device = torch.device("cpu")
        self._max_length = 2048
        self.batch_size_per_gpu = 1
        self.logits_cache = False
        self.normalize_log_probs = normalize_log_probs
        self.custom_prefix_token_id = 0
        self.logits = torch.tensor([1.234, -0.876, 0.345], dtype=dtype)

    def _model_call(self, inps, **kwargs):
        return self.logits.expand(inps.shape[0], inps.shape[1], -1)

    def tok_encode(self, string):
        return [1] * int(string)


class TestLikelihoodPrecision(unittest.TestCase):
    def test_long_continuation_matches_high_precision_reference(self):
        # Include both orders of magnitude so reordering/padding cannot hide a
        # precision regression. References use the model's already rounded logits.
        requests = [(("short", "choice"), [0], [2] * 7),
                    (("document", "continuation"), [0], [1] * 1001)]
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for normalize in (False, True):
                with self.subTest(dtype=dtype, normalize=normalize):
                    lm = FixedLogitsHFLM(dtype, normalize)
                    actual = lm._loglikelihood_tokens(requests, disable_tqdm=True)
                    log_probs = torch.log_softmax(lm.logits.double(), dim=-1)
                    for (_, _, continuation), (score, greedy) in zip(requests, actual):
                        selected = log_probs[continuation]
                        reference = selected.mean() if normalize else selected.sum()
                        torch.testing.assert_close(
                            torch.tensor(score), reference.float(), rtol=1e-6, atol=1e-6
                        )
                        self.assertFalse(greedy)

    def test_rolling_document_across_multiple_windows(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                lm = FixedLogitsHFLM(dtype, normalize_log_probs=False)
                token_count = 3073
                actual = lm.loglikelihood_rolling(
                    [SimpleNamespace(args=(str(token_count),))], disable_tqdm=True
                )[0]
                reference = torch.log_softmax(lm.logits.double(), dim=-1)[1] * token_count
                torch.testing.assert_close(
                    torch.tensor(actual), reference.float(), rtol=1e-6, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
