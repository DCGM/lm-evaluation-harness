import collections
import fnmatch
import gc
import itertools
import time
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import transformers
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from lm_eval.utils import eval_logger, SegmentedString


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.configuration_utils import PretrainedConfig


def chunks(iter, n: int = 0, fn=None):
    """
    Divides an iterable into chunks of specified size or based on a given function.
    Useful for batching

    Parameters:
    - iter: The input iterable to be divided into chunks.
    - n: An integer representing the size of each chunk. Default is 0.
    - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

    Returns:
    An iterator that yields chunks of the input iterable.

    Example usage:
    ```
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for chunk in chunks(data, 3):
        print(chunk)
    ```
    Output:
    ```
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10]
    ```
    """
    arr = []
    for i, x in enumerate(iter):
        arr.append(x)
        if len(arr) == (fn(i, iter) if fn else n):
            yield arr
            arr = []

    if arr:
        yield arr


class MultiChoice:
    def __init__(self, choices) -> None:
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values) -> bool:
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                eval_logger.info("Available tasks to choose:")
                for choice in self.choices:
                    eval_logger.info(f"  - {choice}")
                raise ValueError("'{}' is not in task list".format(value))
        return True

    def __iter__(self) -> Iterator:
        for choice in self.choices:
            yield choice


class Grouper:
    """
    takes an array `arr` and function `fn` and returns a dictionary
    with keys fn(ob) for each ob in `arr` and with values `self.arr[key]` a list of all
    objects in `arr` satisfying `key == fn(ob)`.
    """

    def __init__(self, arr, fn) -> None:
        # self.orig_arr = arr
        self.size = len(arr)
        arr = list(enumerate(arr))

        def group_return_dict(arr, fn):
            res = collections.defaultdict(list)

            for ob in arr:
                res[fn(ob)].append(ob)
            return res

        arr = group_return_dict(arr, lambda x: fn(x[1]))

        # self.arr has format Dict[Tuple[int, <entry from orig. arr>]]
        self.arr = arr
        self._grouped = None

    def get_grouped(self):
        # return the contents but not indices for our grouped dict.
        if self._grouped:
            return self._grouped
        grouped = {}
        for key in self.arr.keys():
            # drop the index from each element of self.arr
            grouped[key] = [y[1] for y in self.arr[key]]
        self._grouped = grouped
        return grouped

    def get_original(self, grouped_dict):
        # take in a grouped dictionary with e.g. results for each key listed
        # in the same order as the instances in `self.arr`, and
        # return the results in the same (single list) order as `self.orig_arr`.
        res = [None] * self.size
        cov = [False] * self.size
        # orig = [None] * self.size

        assert grouped_dict.keys() == self.arr.keys()

        for key in grouped_dict.keys():
            for (ind, _), v in zip(self.arr[key], grouped_dict[key]):
                res[ind] = v
                cov[ind] = True
                # orig[ind] = _

        assert all(cov)
        # assert orig == self.orig_arr

        return res


def pad_and_concat(
    max_length: int,
    tensors: List[torch.Tensor],
    padding_side: Literal["right", "left"] = "right",
):
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    assert (
        padding_side == "left" or padding_side == "right"
    ), f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = torch.cat(
                    [
                        tensor,  # [seq]
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
            else:
                # left-pad
                tensors[i] = torch.cat(
                    [
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)

    return torch.cat(tensors, dim=0)


def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def undistribute(iterable):
    """
    Undoes https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute .

    Re-interleaves results that have been split using more_itertools.distribute:
        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]
        >>> undistribute([group_1, group_2])
        [1, 2, 3, 4, 5, 6]

    Handles non-uniform component lengths:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]
        >>> undistribute(children)
        [1, 2, 3, 4, 5, 6, 7]

    Also handles when some iterables are empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]
        >>> undistribute(children)
        [1, 2, 3]

    """

    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(*[list(x) for x in iterable])
        )
        if x is not None
    ]


def retry_on_specific_exceptions(
    on_exceptions: List[Type[Exception]],
    max_retries: Optional[int] = None,
    backoff_time: float = 3.0,
    backoff_multiplier: float = 1.5,
    on_exception_callback: Optional[Callable[[Exception, float], Any]] = None,
):
    """Retry on an LLM Provider's rate limit error with exponential backoff
    For example, to use for OpenAI, do the following:
    ```
    from openai import RateLimitError

    # Recommend specifying max_retries to avoid infinite loops!
    @retry_on_specific_exceptions([RateLimitError], max_retries=3)
    def completion(...):
        # Wrap OpenAI completion function here
        ...
    ```
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep_time = backoff_time
            attempt = 0
            while max_retries is None or attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except tuple(on_exceptions) as e:
                    if on_exception_callback is not None:
                        on_exception_callback(e, sleep_time)
                    time.sleep(sleep_time)
                    sleep_time *= backoff_multiplier
                    attempt += 1

        return wrapper

    return decorator


class Collator:
    """
    A class for reordering and batching elements of an array.

    This class allows for sorting an array based on a provided sorting function, grouping elements based on a grouping function, and generating batches from the sorted and grouped data.

    Objects of this class have the group_by attribute which determines the method for grouping
    the data while batching it. Three options include "gen_kwargs", "contexts", or None:
        If group_by == "gen_kwargs" then requests will be grouped by gen_kwargs
        If group_by == "contexts" then requests will be grouped by context + cont[:-1]
        If None then requests will just be reordered by length descending.
    """

    def __init__(
        self,
        arr: List,
        sort_fn: Callable = lambda x: x,
        group_fn: Callable = lambda x: x[1],
        group_by: Union[Literal["gen_kwargs", "contexts"], None] = None,
    ) -> None:
        self._group_by = group_by
        # 0 indices are enumerated indices. Apply functions to original arr.
        self._sort_fn = lambda x: sort_fn(x[1])
        self._group_fn = lambda x: group_fn(x[1])
        self._reorder_indices: List = []
        self._size = len(arr)
        self._arr_with_indices: Union[Dict, Tuple[Tuple[int, Any], ...]] = tuple(
            enumerate(arr)
        )  # [indices, (arr)]
        if self._group_by == "contexts":
            self._group_by_context()
        elif self._group_by == "gen_kwargs":
            self._group_by_index()

    def _group_by_index(self) -> None:
        """Group the elements of a list based on their indices."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="gen_kwargs"
        )

    def _group_by_context(self) -> None:
        """Group the array with indices by context."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="contexts"
        )

    def get_batched(self, n: int = 1, batch_fn: Optional[Callable] = None) -> Iterator:
        """
        Generates and yields batches from the reordered array. The method of grouping and batching
        depends on the parameter `group_by`.
        If `group_by` is set to "gen_kwargs", it will batch the
        re-ordered values with same gen_kwargs for each batch.
        If `group_by` is "contexts", it caches the requests by context before batching.
        If `group_by` is neither "gen_kwargs" nor "contexts", it yields the reordered array

        Parameters:
        - n (int): The size of each batch. Defaults to 1.
        - batch_fn ([Callable[[int, Iterable], int]] | None): A function to determine the size of
          each batch. Optional, defaults to None.

        Returns:
        Iterator: An iterator over batches of reordered elements grouped as per the `group_by`
                  attribute.

        Yields:
        List of batched elements according to the `group_by` attribute.
        """
        if self._group_by == "gen_kwargs":
            for (
                key,
                values,
            ) in self._arr_with_indices.items():  # type: ignore
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        elif self._group_by == "contexts":
            # Get one sample from each key
            values = self._reorder(
                [value[0] for value in self._arr_with_indices.values()]
            )
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch
        else:
            values = self._reorder(self._arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def get_cache(
        self,
        req_str: Tuple[str, str] = None,
        cxt_toks: List[int] = None,
        cont_toks: List[int] = None,
        logits: torch.Tensor = None,
    ) -> Iterator[Tuple[Tuple[str, str], List[int], torch.Tensor]]:
        """
        Retrieves cached single-token continuations and their associated arguments, updating indices as necessary.

        The behavior of this function varies depending on how the `group_by` attribute is set:

        - When `group_by` is "contexts":
            The function identifies single-token continuations by checking for keys that equate to
            [context+continuation][-1] and logs the indices for re-ordering.
            In this mode, this function can work in two scenarios:

            1. Cache Hit - Single Match:
                If a single matching context-continuation pair is found in the cache,
                the function yields the original arguments.

            2. Cache Hit - Multiple Matches:
                If multiple matching context-continuation pairs are found in the cache,
                the function expands the logits batch dimension to match the number of cache hits.
                It updates the original requests and continuation tokens.

        - When `group_by` is not set to "contexts":
            This method yields the original arguments, logits and continuation tokens,
            without checking for one-token continuations.

        Parameters:
        - req_str (tuple[str, str]): Original strings used for CachingLM.
        - cxt_toks (list[int]): Full context tokens used for lookup.
        - cont_toks (list[int]): Continuation tokens for which logits were generated.
        - logits (torch.Tensor [1, seq_length, vocab_size]): Logits generated by the model given context and continuation keys.

        Yields:
        - Iterator:
            - req_str (tuple[str, str]): strings used for CachingLM.
            - cont_toks (list[int]) : continuation tokens.
            - logits (torch.Tensor [1, seq_length, vocab_size]): The original logits (repeated cache hit times)
        """
        if self._group_by == "contexts":
            cache_hit: List[
                Tuple[int, Tuple[Tuple[str, str], List[int], List[int]]]
            ] = self._arr_with_indices.pop(tuple(cxt_toks + cont_toks[:-1]))
            if (cache_size := len(cache_hit)) == 1:
                self._reorder_indices.extend(x[0] for x in cache_hit)
                yield req_str, cont_toks, logits
            else:
                # If we have matching requests then expand the batch dimension (no-op) and
                # yield each along with its corresponding args.
                multilogits = logits.expand(cache_size, -1, -1).chunk(cache_size)
                indices, req_str, cont_toks = zip(
                    *[(x[0], x[1][0], x[-1][-1]) for x in cache_hit]
                )
                self._reorder_indices.extend(indices)
                for c_key, cont_tok, logit in zip(req_str, cont_toks, multilogits):
                    yield c_key, cont_tok, logit
        else:
            yield req_str, cont_toks, logits

    def _reorder(self, arr: Union[List, Tuple[Tuple[int, Any], ...]]) -> Iterator:
        """
        Reorders the elements in the array based on the sorting function.

        Parameters:
        - arr (list | tuple[tuple[int, Any], ...]]): The array or iterable to be reordered.

        Yields:
            Iterator
        """
        arr = sorted(arr, key=self._sort_fn)
        if not self._group_by == "contexts":
            # If grouped by contexts then indices will be set in get_cache()
            self._reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: List) -> List:
        """
        Restores the original order of elements from the reordered list.

        Parameters:
        - newarr (list): The reordered array.

        Returns:
        list: The array with elements restored to their original order.
        """
        res = [None] * self._size
        cov = [False] * self._size

        for ind, v in zip(self._reorder_indices, newarr):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res

    def __len__(self):
        return self._size

    @staticmethod
    def group(
        arr: Iterable,
        fn: Callable,
        group_by: Literal["gen_kwargs", "contexts"] = "gen_kwargs",
    ) -> dict:
        """
        Groups elements of an iterable based on a provided function.


        The `group_by` parameter determines the method of grouping.
        If `group_by` is "contexts", the elements are grouped by [context + cont][:-1].
        If `group_by` is "gen_kwargs", the elements are grouped based on the gen_kwargs dict.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function to determine the grouping.
        - values (bool): If True, returns the values of the group. Defaults to False.

        Returns:
        Iterator: An iterable of grouped elements.
        """
        res = collections.defaultdict(list)
        for ob in arr:
            # where ob == [context + cont]
            if group_by == "contexts":
                res[tuple(fn(ob))].append(ob)
            else:
                try:
                    hashable_dict = tuple(
                        (
                            key,
                            tuple(value)
                            if isinstance(value, collections.abc.Iterable)
                            else value,
                        )
                        for key, value in sorted(fn(ob).items())
                    )
                    res[hashable_dict].append(ob)
                except (TypeError, AttributeError):
                    res[tuple(fn(ob))].append(ob)
        return res

    @staticmethod
    def get_chunks(_iter, n: int = 0, fn=None):
        """
        Divides an iterable into chunks of specified size or based on a given function.
        Useful for batching

        Parameters:
        - iter: The input iterable to be divided into chunks.
        - n: An integer representing the size of each chunk. Default is 0.
        - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

        Returns:
        An iterator that yields chunks of the input iterable.

        Example usage:
        ```
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for chunk in chunks(data, 3):
            print(chunk)
        ```
        Output:
        ```
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
        ```
        """
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr


def configure_pad_token(
    tokenizer: "PreTrainedTokenizerBase",
    model_config: Optional["PretrainedConfig"] = None,
) -> "PreTrainedTokenizerBase":
    """
    This function checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.
    Some tokenizers require special handling.

    Args:
        tokenizer: The tokenizer for which the padding token is to be handled.
        model_config: The configuration of the model. Default is None.

    Returns:
        The tokenizer after the padding token has been handled.

    Raises:
        AssertionError: If the tokenizer is of type RWKVWorldTokenizer or Rwkv5Tokenizer and the padding token id is not 0.
    """
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # handle special cases
        if model_config and getattr(model_config, "model_type", None) == "qwen":
            # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
            tokenizer.pad_token = "<|endoftext|>"
        elif (
            tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
            or tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
        ):
            # The RWKV world tokenizer, does not allow for adding special tokens / setting the pad token (which is set as 0)
            # The additional tokenizer name check is needed, as there exists rwkv4 models with neox tokenizer
            # ---
            # Note that the world tokenizer class name, might change in the future for the final huggingface merge
            # https://github.com/huggingface/transformers/pull/26963
            assert tokenizer.pad_token_id == 0
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    return tokenizer


def truncate_token_segments_from_left(segments: List[List[int]], max_length: int) -> List[List[int]]:
    """
    Truncates a list of token segments from the left to the specified maximum length.

    Parameters:
    - segments (List[List[int]]): The list of token segments to be truncated.
    - max_length (int): The maximum length of the segments.

    Returns:
    List[List[int]]: The list of token segments truncated to the specified maximum length.
    """
    assert max_length > 0, "max_length must be greater than 0"
    new_segmented_tokens = []
    new_length = 0
    for segment in reversed(segments):
        new_length += len(segment)
        if new_length <= max_length:  # we can add the whole segment
            new_segmented_tokens.append(segment)
        elif new_length - len(segment) < max_length:  # we can add part of the segment
            rest_of_space = max_length - (new_length - len(segment))
            new_segmented_tokens.append(segment[-rest_of_space:])
            break
        else:
            break

    return list(reversed(new_segmented_tokens))


def truncate_token_segments_from_right(segments: List[List[int]], max_length: int) -> List[List[int]]:
    """
    Truncates a list of token segments from the right to the specified maximum length.

    Parameters:
    - segments (List[List[int]]): The list of token segments to be truncated.
    - max_length (int): The maximum length of the segments.

    Returns:
    List[List[int]]: The list of token segments truncated to the specified maximum length.
    """
    assert max_length > 0, "max_length must be greater than 0"
    new_segmented_tokens = []
    new_length = 0
    for segment in segments:
        new_length += len(segment)
        if new_length <= max_length:  # we can add the whole segment
            new_segmented_tokens.append(segment)
        elif new_length - len(segment) < max_length:  # we can add part of the segment
            rest_of_space = max_length - (new_length - len(segment))
            new_segmented_tokens.append(segment[:rest_of_space])
            break
        else:
            break

    return new_segmented_tokens


def segmented_tok_encode(string: SegmentedString, tokenizer: PreTrainedTokenizerFast, max_length: int,
                         truncate_strategy: Optional[str], add_special_tokens=False) -> Tuple[
    List[int], List[List[int]], Optional[List[str]]]:
    """
    Tokenizes a string and that splits it into a segments. It is using offsets mapping thus it is only compatible
    with fast tokenizers.

    Args:
        string: SegmentedString
            The string to be tokenized.
        tokenizer: PreTrainedTokenizerFast
            The tokenizer to be used.
        max_length: int
            The maximum number of output tokens.
        truncate_strategy: Optional[str]
            The strategy to use when truncating the string. If None, the string will not be truncated.
            If "to_max_length", the string will be truncated to the max_length.
            If "leave_description", the string will be truncated to the max_length, but the description will be kept
            in full.
        add_special_tokens: bool
            Whether to add special tokens to the tokenized string.

    Returns:
        Tuple[List[int], List[List[int]], Optional[List[str]]]:
        Flat list of tokens and list of lists of tokens, each list corresponding to a segment. The third element
        is a list of segment labels.

    """
    if type(string)!=SegmentedString:
        raise ValueError(f"Input must be of type SegmentedString (found type {str(type(string))}).\n"
                         f"Do not use smart truncation strategy for language modeling tasks.")
    assert type(string) == SegmentedString, "string must be a SegmentedString"
    encoding = tokenizer(
        string,
        add_special_tokens=add_special_tokens,
        truncation=False,
        return_offsets_mapping=True
    )
    token_offsets = encoding["offset_mapping"]
    encoding = encoding["input_ids"]

    segment_offset = 0
    offset = len(string.segments[segment_offset])

    segmented_tokens = [[]]

    segment_labels = [string.labels[segment_offset]] if string.labels else None

    has_continuation = "target_cont" in string.labels

    # collect tokens for each segment, tokens with start >= length of the segment are in the next segment
    for token, tok_offsets in zip(encoding, token_offsets):
        if tok_offsets[0] >= offset:  # if current token's start offset is greater than the current segment's offset
            segment_offset += 1  # move to the next (new) segment
            offset += len(
                string.segments[segment_offset])  # add char offset corresponding to the length of the new segment

            segmented_tokens.append([])
            if segment_labels is not None:
                segment_labels.append(string.labels[segment_offset])
        segmented_tokens[-1].append(token)
    # the last segmment might be placed into delimiter, make it cont instead
    if segment_labels is not None and segment_labels[-1] in ["target_delimiter", "target_text"] and has_continuation:
        segment_labels[-1] = "target_cont"

    if len(encoding) > max_length:
        if truncate_strategy == "leave_description":
            # prefix must have at least 20% of the tokens
            min_prefix_length = max_length // 5

            try:
                desc_pos = string.labels.index("description")
            except ValueError as e:
                print(f"Description not found in {str(string)}\n{str(string.labels)}")
                raise e

            # we take everything before and the description itself, then we will truncate the rest
            desc_tokens = segmented_tokens[desc_pos]
            desc_prefix = [t for tokens in segmented_tokens[:desc_pos] for t in tokens]
            desc_budget = len(desc_prefix) + len(desc_tokens)

            # compute size of
            if desc_budget > max_length:
                raise ValueError(f"{tokenizer.decode(desc_prefix + desc_tokens, skip_special_tokens=False)}\n"
                                 f"Description tokens: {desc_budget}. Description is too long for max length {max_length}!")
            else:
                """
                In chat template, there might be already something before description,
                solution: do not truncate that either!
                """
                proposed_encoding = desc_prefix + desc_tokens + encoding[desc_budget:][-(max_length - desc_budget):]
                truncated_rest = truncate_token_segments_from_left(segmented_tokens[desc_pos + 1:],
                                                                   max_length - desc_budget)
                proposed_segmented_tokens = segmented_tokens[:desc_pos + 1] + truncated_rest
                proposed_segment_labels = segment_labels[:desc_pos + 1] + segment_labels[-len(truncated_rest):]
                if 'target_text' not in proposed_segment_labels:
                    if has_continuation:
                        target_cont_pos = segment_labels.index("target_cont")
                        split_pos = target_cont_pos - 1 if segment_labels[
                                                           target_cont_pos - 1] == "target_delimiter" else target_cont_pos
                    else:
                        split_pos = len(segment_labels)


                    target_prefix_positions = list(
                        range(segment_labels.index("target_text"), split_pos))
                    target_prefix_tokens = [segmented_tokens[i] for i in target_prefix_positions]
                    target_prefix_labels = [segment_labels[i] for i in target_prefix_positions]

                    target_suffix_positions = list(
                        range(split_pos, len(segment_labels)))
                    target_suffix_tokens = [segmented_tokens[i] for i in target_suffix_positions]
                    target_suffix_labels = [segment_labels[i] for i in target_suffix_positions]

                    # truncate  suffix from right
                    truncated_segments_suffix = truncate_token_segments_from_right(target_suffix_tokens,
                                                                                   max_length - desc_budget - min_prefix_length)
                    # truncate target text frm left
                    truncated_segments_prefix = truncate_token_segments_from_left(target_prefix_tokens,
                                                                                  min_prefix_length)
                    #  get fixed encoding
                    encoding = desc_prefix + desc_tokens + \
                               ([t for tokens in truncated_segments_prefix for t in tokens] +
                                [t for tokens in truncated_segments_suffix for t in tokens])
                    # get fixed segmented tokens
                    segmented_tokens = segmented_tokens[
                                       :desc_pos + 1] + truncated_segments_prefix + truncated_segments_suffix
                    # get fixed segment labels
                    segment_labels = (segment_labels[:desc_pos + 1] +
                                      target_prefix_labels[-len(truncated_segments_prefix):] +
                                      target_suffix_labels[:len(truncated_segments_suffix)])

                    # this assumes that in this case, chat template segments before target text are dropped
                    # only chat template segment left is the one before description

                    assert len(segment_labels) == len(segmented_tokens)

                else:
                    encoding = proposed_encoding
                    segmented_tokens = proposed_segmented_tokens
                    segment_labels = proposed_segment_labels

        elif truncate_strategy == "to_max_length":
            encoding = encoding[-max_length:]
            segmented_tokens = truncate_token_segments_from_left(segmented_tokens, max_length)
            if segment_labels is not None:
                segment_labels = segment_labels[-len(segmented_tokens):]
    if has_continuation and  "target_cont" not in segment_labels:
        raise ValueError(f"Continuation not found in {str(string)}\n{str(segment_labels)}")
    return encoding, segmented_tokens, segment_labels
