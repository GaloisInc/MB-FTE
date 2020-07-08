from abc import ABC, abstractmethod
from enum import Enum
import operator
from time import time
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Tuple
import numpy as np
from mbfte import _logger


class Framework(Enum):
    """
    The supported machine learning frameworks.
    """

    PYTORCH = 1


def get_predictions(
    logits: Any,
    model: str = "pytorch",
    avoid_tokens: bool = True,
) -> List[Tuple[int, float]]:
    """Get model predictions."""
    index: int
    if model == "pytorch":
        index = -1
    else:
        ValueError(f"Invalid model '{model}'")
    # logits should be a 2D array where for each token in the input (in order), we
    # have a score for each potential next token. We only care about the scoring
    # associated with the last token in the input stream.
    if avoid_tokens:
        for i in TOKENS_TO_AVOID:
            logits[index][i] = -10000
    v = softmax(logits[index])
    nonzeros = np.argwhere(v)
    return [(x[0], v[x[0]]) for x in nonzeros]


def softmax(input: List[float]) -> List[float]:
    ex = np.exp(input - np.max(input))
    result = ex / ex.sum(axis=0)
    if TYPE_CHECKING:
        assert isinstance(result, list)
    return result


class AbstractModelWrapper(ABC):
    """An abstract class for wrapping a GPT-2 model for use in MB-FTE."""

    NAME: str

    @abstractmethod
    def __init__(
        self,
        model_dir: str,
        temperature: float,
    ) -> None:
        """Initialize the model at `model_dir` using the provided
        `temperature`."""
        pass

    @abstractmethod
    def get_token(self, index: int) -> str:
        """Return the token associated with the given token `index`."""
        pass

    @abstractmethod
    def tokenize(self, sentence: str) -> List[int]:
        """Tokenize the given `sentence` into a list of token indices."""
        pass

    @abstractmethod
    def prediction(
        self,
        sentence: str,
        past: Optional[Any] = None,
        first: bool = False,
        save_past: bool = True,
    ) -> Tuple[List[Any], Any]:
        """Produce a prediction from the model.

        Args:
         sentence:
            The prior token to use for the prediction.
         past:
            Any past state to use.
         first:
            Whether we're making our first prediction or not.
        """
        pass

    def cumprob_gen(
        self,
        input: str,
        seed: str,
        save_past: bool = False,
    ) -> Generator[Tuple[float, float], None, List[int]]:
        """
        A generator for extracting the cumulative probabilities for some input
        string.

        Args:
            input (str): The input string.
            seed (str): The model seed to use.
            save_past (bool): Whether to save the past on each model prediction or not.

        Returns:
            List[int]: A list of token indices chosen, where the token index
            corresponds to the cumulative probability (that is, a token index of
            0 means the most likely token, _not_ the token corresponding to the
            first entry in the token list).

        Yields:
            Generator[Tuple[float, float], None, None]: The cumulative
            probability of the current token and the cumulative probability of
            the previous token.
        """
        t0 = time()
        token_indices: List[Any] = self.tokenize(input)
        t1 = time()
        _logger.info(f"`tokenize`: {t1 - t0:.4f}s")
        # A list containing the token indices corresponding to the cumulative
        # probability.
        cumprob_token_indices: List[int] = []

        full: str = seed
        last_token: Optional[str] = None
        past: Optional[Any] = None
        for i, token_index in enumerate(token_indices):
            # Get predicitions
            t0 = time()
            if i == 0:
                pred, past = self.prediction(seed, past=None, first=True)
            else:
                assert last_token is not None
                full += last_token
                sentence: str = last_token if save_past else full
                pred, past = self.prediction(
                    sentence, past, first=False, save_past=save_past
                )
            # `pred` contains the following: `[token index, token probability]`.
            # We sort `pred` based on token probability so we can find the token
            # associated with the cumulative probability.
            pred.sort(key=operator.itemgetter(1), reverse=True)
            # The last token's cumulative probability.
            last: float = 0.0
            # The cumulative probability and prior cumulative probability for
            # the chosen token.
            result: Optional[Tuple[float, float]] = None
            for j in range(len(pred)):
                pred_token_index, probability = pred[j]
                token = self.get_token(pred_token_index)
                # The current token's cumulative probability.
                current: float = 0.0
                if j == len(pred) - 1:
                    # Make sure the last token has a probability of 1.
                    current = 1.0
                else:
                    current = probability + last

                if pred_token_index == token_index:
                    # We found a match! Save any necessary info and return the
                    # associated cumulative probability info for this index.
                    last_token = token
                    result = (current, last)
                    cumprob_token_indices.append(j)
                    break
                else:
                    last = current

            t1 = time()
            _logger.info(f"Iteration {i}: {t1 - t0:.4f}s")

            assert result is not None
            yield result

        return cumprob_token_indices

    def cumprob_indices(
        self, input: str, seed: str, save_past: bool = False
    ) -> List[int]:
        """
        Compute the cumulative probability indices for an input string.

        Args:
            input (str): The input string.
            seed (str): The model seed to use.

        Returns:
            List[int]: The resulting cumulative probability indices.
        """
        generator = self.cumprob_gen(input, seed, save_past)
        try:
            while True:
                next(generator)
        except StopIteration as result:
            return result.value  # type: ignore

    def cumprob_from_input_and_past(
        self,
        input: str,
        past: Optional[Any],
        use_past: bool = True,
    ) -> Tuple[List[Tuple[int, float]], Any]:
        """
        Compute the cumulative probability distribution from a given input
        string and past model state.

        Args:
            input (str): The input string to use.

            past (Optional[Any]): The past model state, if any.

            use_past (bool, optional): Whether to use the past state or not.
            Defaults to True.

        Returns:
            Tuple[List[Tuple[int, float]], Any]: The cumulative probabillity
            distribution alongside an updated model state.
        """
        if past is None:
            probs, past = self.prediction(input, past=None, first=True)
        else:
            probs, past = self.prediction(input, past, first=False, save_past=use_past)
        return (_cumulative_probabilities(probs), past)


def _cumulative_probabilities(
    probabilities: List[Tuple[int, float]]
) -> List[Tuple[int, float]]:
    """Return a list of `(token index, cumulative probability)` tuples
    sorted from lowest cumulative probability to highest."""
    probabilities.sort(key=operator.itemgetter(1), reverse=True)
    cumulative_probs: List[Tuple[int, float]] = []
    for i, (index, probability) in enumerate(probabilities):
        cumprob: float
        if i == 0:
            cumprob = probability
        elif i == len(probabilities) - 1:
            cumprob = 1.0
        else:
            cumprob = cumulative_probs[i - 1][1] + probability
        cumulative_probs.append((index, cumprob))
    return cumulative_probs


TOKENS_TO_AVOID: List[int] = [
    50256,
    220,
    628,
    31,
    2488,
    4275,
    12404,
    22675,
    25248,
    32406,
    37991,
    41573,
    44212,
    48193,
    2234,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    447,
    564,
    1209,
    1587,
    1792,
    2343,
    2515,
    4204,
    4210,
    5008,
    5099,
    5525,
    6184,
    6353,
    6408,
    6552,
    7134,
    7377,
    8008,
    8582,
    8955,
    10253,
    10263,
    10310,
    10545,
    11019,
    11737,
    11805,
    11976,
    12045,
    12100,
    12466,
    12520,
    12859,
    13305,
    13328,
    13783,
    13945,
    14360,
    14519,
    14524,
    15139,
    15474,
    15926,
    16268,
    17312,
    17358,
    17433,
    17550,
    17683,
    17739,
    17804,
    17992,
    18004,
    18074,
    18433,
    18796,
    18872,
    18923,
    19021,
    19049,
    19469,
    19526,
    19567,
    20015,
    20046,
    20174,
    20543,
    20724,
    20998,
    21253,
    22522,
    22755,
    22757,
    22880,
    22887,
    23294,
    23329,
    23596,
    23626,
    23821,
    23877,
    24231,
    24583,
    24861,
    24966,
    25001,
    25081,
    25370,
    25443,
    26193,
    26292,
    26344,
    26486,
    26534,
    26825,
    27032,
    27332,
    27670,
    27764,
    27950,
    28053,
    28156,
    28225,
    28839,
    28938,
    29705,
    29773,
    29785,
    29826,
    30266,
    30298,
    30325,
    30585,
    31204,
    31479,
    31619,
    31965,
    32003,
    32368,
    32391,
    32432,
    32518,
    32573,
    32849,
    33176,
    33232,
    33426,
    33434,
    33566,
    33699,
    33768,
    33951,
    34247,
    34402,
    34460,
    34504,
    34650,
    34719,
    34754,
    34932,
    35050,
    35069,
    35266,
    35705,
    35707,
    35975,
    36181,
    36365,
    36469,
    36596,
    36685,
    37239,
    37345,
    37389,
    37605,
    37772,
    37863,
    38184,
    38461,
    39333,
    39355,
    39374,
    39611,
    39820,
    40367,
    40670,
    41340,
    41365,
    41585,
    41678,
    41753,
    41840,
    42062,
    42164,
    42314,
    42527,
    43074,
    43102,
    43297,
    43380,
    43518,
    43636,
    43718,
    43769,
    43889,
    43897,
    44165,
    44293,
    45250,
    45379,
    45433,
    45495,
    45539,
    45617,
    45739,
    45784,
    45865,
    45911,
    46237,
    46256,
    46349,
    46479,
    46695,
    46763,
    46788,
    47078,
    47249,
    47490,
    47703,
    47728,
    47797,
    47947,
    47991,
    48071,
    48585,
    48953,
    48958,
    49035,
    49149,
    49426,
    49694,
    50159,
    50169,
]
"""The token indices to avoid.

The token indices correspond to the indices in the vocab lists of either
`pytorch` (the `gpt2-vocab.json` file) or `tf` (the `encoder.json` file). Note
that both vocab lists are the same."""
