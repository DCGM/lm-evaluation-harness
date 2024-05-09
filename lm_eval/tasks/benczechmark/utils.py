# -*- coding: UTF-8 -*-
"""

:authors:     Martin Dočekal, Martin Fajčík
"""
from typing import List, Tuple
from typing import Optional

import datasets
import evaluate
import numpy
from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix

from lm_eval.api.metrics import mean
from lm_eval.api.task import ConfigurableTask, eval_logger


###
### F1 metric
###

# The f1_posterior and _evaluate_statistics implementation is based on [GOUTTE-2005], and these few lines were borrowed
# and modified from Andre Anjos <anjos@idiap.ch> under Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
def f1_posterior(tp, fp, fn, lambda_, nb_samples):
    """Simulates the F1-score posterior of a system with the provided markings

    This implementation is based on [GOUTTE-2005]_, equation 11.

    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA "false alarm", or "Type I error"

    fn : int
        False Negative count, AKA "miss", or "Type II error"

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.  If unsure, use 0.5.

    nb_samples : int
        number of generated gamma distribution values


    Returns
    -------

    variates : numpy.ndarray
        An array with size ``nb_samples`` containing a realization of equation
        11.

    """

    u = numpy.random.gamma(shape=(tp + lambda_), scale=2.0, size=nb_samples)
    v = numpy.random.gamma(
        shape=(fp + fn + (2 * lambda_)), scale=1.0, size=nb_samples
    )
    return u / (u + v)


def _evaluate_statistics(variates, coverage):
    """Evaluates the left and right margins for a given M-C distribution


    Parameters
    ----------

    variates : numpy.ndarray
        A 1-D array containing the simulated variates

    coverage : float
        A number, between 0 and 1 to indicate the desired coverage.  Typically,
        this number is set to 0.95 (95% coverage).


    Returns
    -------

    stats : (float, float, float, float)
        mean, mode and credible intervals for the input simulation

    """

    left_half = (1 - coverage) / 2  # size of excluded (half) area
    sorted_variates = numpy.sort(variates)

    # n.b.: we return the equally tailed range

    # calculates position of score which would exclude the left_half (left)
    lower_index = int(round(len(variates) * left_half))

    # calculates position of score which would exclude the right_half (right)
    upper_index = int(round(len(variates) * (1 - left_half)))

    lower = sorted_variates[lower_index - 1]
    upper = sorted_variates[upper_index - 1]

    return lower, upper


def aggregate_macro_f1_score(items, **kwargs):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average='macro')
    return fscore


def aggregate_macro_f1_CI(items, alpha=0.95):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]

    # Calculate confusion matrix
    cm = confusion_matrix(golds, preds)

    # Get unique labels
    unique_labels = numpy.unique(golds + preds)

    # Iterate over confusion matrix to compute metrics for each class
    samples = []
    for i in range(len(unique_labels)):
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP

        # get samples from binary F1 distribution
        samples.append(f1_posterior(TP, FP, FN, 1, 10000))  # 1 = flat prior

    # convert binary f1 samples to macro f1 samples
    samples = numpy.array(samples)
    samples = numpy.mean(samples, axis=0)

    # estimate the credible interval
    lower, upper = _evaluate_statistics(samples, alpha)

    return (lower, upper)


###
### ROUGE metric
###

def rouge_raw_r1_low_f(predictions, references):
    return rouge_raw(predictions, references, "1_low_fmeasure")


def rouge_raw_r1_mid_f(predictions, references):
    return rouge_raw(predictions, references, "1_mid_fmeasure")


def rouge_raw_r1_high_f(predictions, references):
    return rouge_raw(predictions, references, "1_high_fmeasure")


def rouge_raw_r2_low_f(predictions, references):
    return rouge_raw(predictions, references, "2_low_fmeasure")


def rouge_raw_r2_mid_f(predictions, references):
    return rouge_raw(predictions, references, "2_mid_fmeasure")


def rouge_raw_r2_high_f(predictions, references):
    return rouge_raw(predictions, references, "2_high_fmeasure")


def rouge_raw_rl_low_f(predictions, references):
    return rouge_raw(predictions, references, "L_low_fmeasure")


def rouge_raw_rl_mid_f(predictions, references):
    return rouge_raw(predictions, references, "L_mid_fmeasure")


def rouge_raw_rl_high_f(predictions, references):
    return rouge_raw(predictions, references, "L_high_fmeasure")


def rouge_raw(predictions, references, select: Optional[str] = None):
    module = evaluate.load("CZLC/rouge_raw")
    return module.compute(predictions=predictions, references=references, select=select)


def make_toy_dataset(dataset: datasets.Dataset):
    """
    Makes a toy dataset from given dataset. It means that the dataset will contain only 10 samples.

    :param dataset: Dataset to make toy from.
    """
    return dataset.select(range(10))


###
### MC-AUROC metric
###

def avg_mcauroc(prediction, reference):
    return prediction, reference  # nothing to process, passthrough metric


def aggregate_avg_mcauroc(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    probs = unzipped_list[1]

    metric = evaluate.load("CZLC/mc_auroc")
    result = metric.compute(predictions=probs, references=golds)
    return result["mc_auroc_score"]


def aggregate_CI_avg_mcauroc(items, alpha=0.95):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    probs = unzipped_list[1]

    metric = evaluate.load("CZLC/mc_auroc")
    result = metric.compute(predictions=probs, references=golds)
    return result["mc_auroc_ci"]


###
### Tasks wrapper
###

class MultipleChoiceTask(ConfigurableTask):

    def process_results(self, doc: dict, results: List[Tuple[float, bool]]) -> dict:
        lls, is_greedy = zip(*results)

        # retrieve choices in List[str] form, to compute choice lengths, etc.
        choices = self.doc_to_choice(doc)

        if self.multiple_input:
            gold = self.doc_to_text(doc)
        else:
            gold = self.doc_to_target(doc)

        gold_index_error = False
        if isinstance(gold, list):
            gold = [i if i < len(choices) else -100 for i in gold]
            if -100 in gold:
                gold_index_error = True
        else:
            if isinstance(gold, int):
                gold = gold if gold < len(choices) else -100
            elif isinstance(gold, str):
                gold = choices.index(gold) if gold in choices else -100

            if gold == -100:
                gold_index_error = True

        if gold_index_error:
            eval_logger.warning(
                f"Label index was not in within range of available choices,"
                f"Sample:\n\n{doc}\n\n"
            )
        probs = softmax(list(lls)).tolist()
        pred = numpy.argmax(lls)

        if self.multiple_target:
            acc = 1.0 if pred in gold else 0.0
        else:
            acc = 1.0 if pred == gold else 0.0

        use_metric = list(self._metric_fn_list.keys())

        return {
            **({"acc": acc} if "acc" in use_metric else {}),
            **({"macro_f1": (gold, pred)} if "f1" in use_metric else {}),
            **({"macro_f1_ci": (gold, pred)} if "f1" in use_metric else {}),
            **({"avg_mcauroc": (gold, probs)} if "avg_mcauroc" in use_metric else {}),
            **({"avg_mcauroc_ci": (gold, probs)} if "avg_mcauroc" in use_metric else {}),
        }

    def higher_is_better(self) -> dict:
        return {
            "avg_mcauroc": True,
            "avg_mcauroc_ci": True,
            "macro_f1": True,
            "macro_f1_ci": True,
            "acc": True,
        }

    def aggregation(self) -> dict:
        return {
            "avg_mcauroc": aggregate_avg_mcauroc,
            "avg_mcauroc_ci": aggregate_CI_avg_mcauroc,
            "macro_f1": aggregate_macro_f1_score,
            "macro_f1_ci": aggregate_macro_f1_CI,
            "acc": mean,
        }

def process_docs_cs_nec(dataset):
    def _process_dataset(dataset):
        """
        text: source sentence
        entities: list of selected entities. Each entity contains:
        category_id: string identifier of the entity category
        category_str: human-friendly category name in Czech (verbalizer)
        start: index on which the entity starts in the source sentence
        end: index on which the entity ends in the source sentence
        content: entity content, it was created as text[start:end]
        entity_id: unique entity string identifier
        parent_id: If entity was selected inside another entity (e.g. house number inside address), parent_id is the identifier of the parent entity. None otherwise.

        -> PROCESS INTO ->
        label
        answers
        text
        """
        r = []
        for doc in dataset:
            text = doc["text"]
            for entity in doc["entities"]:
                r.append({
                    "label": entity["category_str"],
                    "answers": [entity["content"]],
                    "text": text,
                })
        return r

    return datasets.arrow_dataset.Dataset.from_list(_process_dataset(dataset))

# MMLU multi-choice style (A....Z)
ANSWER_LETTERS = [chr(ord('A') + i) for i in range(ord('Z') - ord('A') + 1)]

def mmlu_get_choice(dataset):
    choice = [c for c in ANSWER_LETTERS if c in dataset.keys()]
    if len(choice) == 0:
        raise ValueError(f"No answer columns found in dataset")
    return choice

def mmlu_get_answer_index(dataset):
    return ANSWER_LETTERS.index(dataset["correct_answer"])

def mmlu_get_question_text(dataset):
    dataset_answer_keys = mmlu_get_choice(dataset)
    question_text = dataset['question'].strip()
    choices_text = "\n".join(f"{c}. {dataset[c]}" for c in dataset_answer_keys)
    return f"{question_text}\n{choices_text}\nOdpověď:"

def mmlu_get_question_text_umimeto(dataset):
    """
    Umimeto has really bad questions, which require the topic at minimum to make sense
    """
    dataset_answer_keys = mmlu_get_choice(dataset)
    question_text = dataset['question'].strip()
    choices_text = "\n".join(f"{c}. {dataset[c]}" for c in dataset_answer_keys)
    topic = dataset["topic"]
    return f"{topic}: {question_text}\n{choices_text}\nOdpověď:"