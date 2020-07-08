from math import floor
from operator import itemgetter
from os.path import isdir
import logging
from time import time
from fixedint import MutableUInt32
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from mbfte import _logger
from mbfte.crypto import RandomPadding
from mbfte.model_wrapper import AbstractModelWrapper

CODING_BITS: int = 32
"""The number of bits to initially attempt to decode when using variable width
encoding."""
CODING_RANGE: int = 2**CODING_BITS - 1
"""The full integer range for which we try to decode into. Equal to
2**CODING_BITS - 1."""
VARIABLE_POINT_EXTRA_BITS: int = 8
"""The number of extra bits to encode to avoid potential decoding errors while
using variable bitwidth arithmetic encoding."""
FIXED_POINT_EXTRA_BITS: int = 5
"""The number of extra bits to encode to avoid potential decoding errors while
using fixed bitwidth arithmetic encoding."""
N_ZERO_SHIFTS: int = 50
"""The number of zero shifts to look for before giving up during encoding."""


class TooManyZeroShifts(Exception):
    """Raised when too many zero shifts occurred during encoding."""

    pass


class UnableToTokenizeCovertext(Exception):
    """Raised when the encoded covertext does not tokenize correctly."""

    pass


class SentinelCheckFailed(Exception):
    """Raised when the sentinel check fails on decoding."""

    pass


class UnableToDecryptValidCiphertext(Exception):
    """Raised when decoding fails to decrypt a ciphertext successfully."""

    pass


class ExtraBitsNotValid(Exception):
    """Raised when the extra bits appended to the ciphertext are invalid."""

    pass


class TextCover:
    """
    Bundles an `AbstractModelWrapper` with a symmetric encryption scheme to
    implement model-based format transforming encryption, where the format is
    text emitted by the `AbstractModelWrapper`.

    All initialization parameters, including model information, `seed`, `key`,
    `temperature`, etc. MUST be shared between the two parties for encoding and
    decoding to work.
    """

    def __init__(
        self,
        model_wrapper: Type[AbstractModelWrapper],
        model_dir: str,
        seed: str,
        key: Optional[bytes] = None,
        temperature: float = 0.8,
        padding: int = 3,
        save_past: bool = True,
        encoding_bitwidth: str = "variable",
    ) -> None:
        """Load a GPT-2 model and prepare an encryption scheme.

        Args:
         model_dir:
            A directory containing the model to use
         seed:
            A prompt for the text generator
         key:
            The symmetric key to use
         temperature:
            Float value controlling randomness in GPT-2 boltzmann distribution.
            Lower temperature results in less random completions. As the
            temperature approaches zero, the model will become deterministic and
            repetitive. Higher temperature results in more random completions.
         padding:
            How much padding to use when encrypting the plaintext during `encode`.
         save_past:
            Whether to re-use the past state of the model.
         encoding_bitwidth:
            Whether to run the arithmetic encoding with variable or fixed bitwidth.
        """
        if key is None:
            from Crypto.Random import get_random_bytes

            key = get_random_bytes(32)

        if not isdir(model_dir) and model_dir != "PRETRAINED":
            raise FileNotFoundError(f"Invalid model directory: {model_dir}")
        t0 = time()
        self.model = model_wrapper(model_dir, temperature)
        t1 = time()
        _logger.info(f"Time to load model: {t1 - t0:.4f}s")
        self._seed = seed
        self._padding = padding
        self.encrypter = RandomPadding(key, bytes_of_padding=self._padding)
        self._save_past = save_past
        self._encdecs: Dict[
            str,
            Tuple[
                Callable[[bytes, bool], str], Callable[[str, bool], Union[bool, str]]
            ],
        ] = {
            "variable": (self.encode_variable_width, self.decode_variable_width),
            "fixed": (self.encode_fixed_width, self.decode_fixed_width),
        }
        # The function to finish decryption, or `None` if decryption hasn't been
        # started yet.
        self.finish_decryption: Optional[Callable[[bytes], Optional[bytes]]] = None

        if encoding_bitwidth not in self._encdecs.keys():
            raise ValueError(
                "Invalid `encoding_bitwidth`, must be one of 'variable' or 'fixed'"
            )
        # For variable width encoding, we need an extra byte of ciphertext for
        # decryption verification to work. TODO: why?
        self._extra_bytes_to_check = 1 if encoding_bitwidth == "variable" else 0
        # For variable width encoding, the last few bits of a ciphertext don't
        # match the expected bonus bits. So we heuristically skip the last few
        # and 🤞 it works.
        self._bonus_bits_to_skip = 8 if encoding_bitwidth == "variable" else 0

        self._arithmetic_encoding_bitwidth = encoding_bitwidth

    def key(self) -> bytes:
        """Return the symmetric key used."""
        return self.encrypter.key()

    def seed(self) -> str:
        """Return the seed text used."""
        return self._seed

    def encode(self, plaintext: str, complete_sentence: bool = True) -> str:
        """Encodes a plaintext.

        Args:
            plaintext: The plaintext message to encode.

            complete_sentence: Whether to try completing the sentence
                when creating the covertext. This uses a simple heuristic
                that checks that the last character in the covertext is
                a '.'.

        Returns:
            str: The encoded covertext.

        Raises:
            TooManyZeroShifts: Unable to encode due to too many zero shifts.
            UnableToTokenizeCovertext: Tokenization failed.

        """
        nonce, ciphertext = self.encrypter.encrypt(bytes(plaintext, encoding="utf-8"))
        _logger.debug(f"Nonce = {nonce!r}")
        _logger.debug(f"Ciphertext = {ciphertext!r}")
        # Combine the `nonce` and `ciphertext` and encode this combined ciphertext.
        ciphertext = nonce + ciphertext

        return self._encdecs[self._arithmetic_encoding_bitwidth][0](
            ciphertext, complete_sentence
        )

    def decode(self, covertext: str) -> str:
        """Decodes a covertext.

        Args:
            covertext: The covertext to decode.

        Returns:
            str: The decoded plaintext message.

        Raises:
            SentinelCheckFailed: The sentinel check failed.
            UnableToDecryptValidCiphertext: Decryption failed.
        """
        result = self._encdecs[self._arithmetic_encoding_bitwidth][1](covertext, False)
        if TYPE_CHECKING:
            assert isinstance(result, str)
        return result

    def check(self, covertext: str) -> bool:
        """Check a potential covertext for a hidden message.

        Args:
            covertext: The covertext to check.

        Returns:
            bool: Whether the check succeeded or not.
        """
        result = self._encdecs[self._arithmetic_encoding_bitwidth][1](covertext, True)
        if TYPE_CHECKING:
            assert isinstance(result, bool)
        return result

    @staticmethod
    def _cumulative_probabilities(
        probabilities: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Return a list of `(token index, cumulative probability)` tuples
        sorted from lowest cumulative probability to highest."""
        probabilities.sort(key=itemgetter(1), reverse=True)
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

    # NOTE: This encoding method is largely deprecated in favor of the
    # fixed-width approach.
    def encode_variable_width(
        self,
        ciphertext: bytes,
        complete_sentence: bool = True,
    ) -> str:
        # NOTE: In the comments below, we use the terminology "encode" to
        # actually denote the arithmetic _decode_ process. This is simply to
        # avoid confusion, as seeing a bunch of "decoding" text in the "encode"
        # procedure might be more confusing!
        ciphertext_bits = self.encrypter.ciphertextbits(ciphertext)
        # The total number of bits we've encoded so far.
        total_encoded: int = 0
        # The low bitrange for encoding (aka arithmetic decoding).
        low: int = 0
        # The high bitrange for encoding (aka arithmetic decoding).
        high: int = CODING_RANGE
        # The ciphertext as an arithmetic encoding. This is constructed
        # iteratively.
        encoded: int = 0
        # The number of bits to try to encode in any given iteration.
        shift: int = CODING_BITS
        # List of token indices chosen. This is used at the end of encoding to check
        # that the produced covertext can be tokenized back to the same token
        # indices.
        token_indices: List[int] = []
        # The last token we've generated.
        last_token: Optional[str] = None
        # List of bits encoded in each iteration. This is used to track if we've hit
        # a setting where we are always encoding zero bits, meaning we're making no
        # progress.
        bits_encoded: List[int] = []
        # The produced covertext.
        covertext: str = ""
        # Contains a list of `(token index, relative probability)` tuples
        # produced during each iteration. This uses `Optional` so we can handle
        # the first iteration differently.
        probs: Optional[List[Tuple[int, float]]] = None
        # How many bits we need to encode. This is used to check when we're
        # done.
        max_bits: int = len(ciphertext) * 8 + VARIABLE_POINT_EXTRA_BITS

        full: str = self._seed

        niters: int = 0
        while True:
            niters += 1
            _logger.info(f"Iteration {niters}")
            # Get prediction probabilities. For the first prediction, we use the
            # provided `seed`. After that, we use the previously generated token.
            t0 = time()
            # NOTE: This `if-else` is the majority of the running time in an
            # iteration.
            if probs is None:
                probs, past = self.model.prediction(self._seed, past=None, first=True)
            else:
                assert last_token is not None
                sentence: str = last_token if self._save_past else full
                probs, past = self.model.prediction(
                    sentence, past, first=False, save_past=self._save_past
                )
            assert probs is not None

            cumulative_probs = TextCover._cumulative_probabilities(probs)

            _logger.debug(f"Desired shift amount: {shift}")

            # Add bits we want to encode according to the shift amount.
            encoded += ciphertext_bits.get(shift)

            _logger.debug(f"Alpha: {low:x} ({low})")
            _logger.debug(f"Beta:  {high:x} ({high})")
            _logger.debug(f"Gamma: {encoded:x} ({encoded})")
            _logger.debug(f"Gamma - Alpha: {encoded - low:x}")

            # Find the first token index with cumulative probability greater
            # than the bits we want to encode.
            i: int = 0
            assert low <= encoded <= high
            bitrange: int = high - low
            while cumulative_probs[i][1] < (encoded - low) / bitrange:
                i += 1

            _logger.debug(f"Upper bound: {(encoded - low) / bitrange}")
            _logger.debug(f"Cumulative probability choice: {i}")
            _logger.debug(f"Selected token index + probability: {cumulative_probs[i]}")

            # Get the token associated with the given token index and add it to
            # `covertext`.
            last_token = self.model.get_token(cumulative_probs[i][0])
            covertext += last_token
            full += last_token
            token_indices.append(cumulative_probs[i][0])

            # Compute the new low and high bit range for encoding the next
            # go-around using the probability range in which we encoded the
            # current token.
            low += floor((cumulative_probs[i - 1][1] if i > 0 else 0.0) * bitrange)
            high -= floor((1 - cumulative_probs[i][1]) * bitrange)

            _logger.debug(f"New low: {low:x} ({low})")
            _logger.debug(f"New high:  {high:x} ({high})")

            # Calculate the number of bits _actually_ encoded.
            shift = 0
            bitrange = high - low
            while bitrange <= CODING_RANGE:
                bitrange *= 2
                shift += 1

            _logger.debug(f"Actual shift amount: {shift}")

            low <<= shift
            high <<= shift
            encoded <<= shift
            total_encoded += shift

            bits_encoded.append(shift)
            if bits_encoded[-N_ZERO_SHIFTS:].count(0) == N_ZERO_SHIFTS:
                _logger.error(f"Encoding failed: Too many zero shifts")
                raise TooManyZeroShifts
            t1 = time()

            _logger.info(f"  Running time: {t1 - t0:.4f}s")
            _logger.debug(f"  Token chosen: {last_token}")
            _logger.debug(
                f"  Number of bits encoded: {shift}. Total: {total_encoded} / {max_bits}"
            )

            if total_encoded >= max_bits:
                if complete_sentence:
                    if covertext[-1] == ".":
                        break
                else:
                    break

        # Encoding fails if tokenizing the covertext doesn't produce the same
        # tokens produced during encoding.
        if self.model.tokenize(covertext) != token_indices:
            _logger.error(f"Encoding failed: Unable to tokenize produced covertext")
            _logger.debug(f"  Covertext:·{covertext}·")
            _logger.debug(f"  Tokenization: {self.model.tokenize(covertext)}")
            _logger.debug(f"  Expected:     {token_indices}")
            raise UnableToTokenizeCovertext

        _logger.info(f"Done encoding! Number of iterations: {niters}")
        return covertext

    # NOTE: This decoding method is largely deprecated in favor of the
    # fixed-width approach.
    def decode_variable_width(
        self,
        covertext: str,
        verify: bool = False,
    ) -> Union[bool, str]:
        # The low bit range for decoding.
        low: int = 0
        # The high bit range for decoding.
        high: int = CODING_RANGE
        # The total number of bits we've encoded so far. This is used to ensure
        # that the integer we end up converting to binary falls on a byte
        # boundary.
        total_encoded: int = 0
        # Reset the "finish decryption" function.
        self.finish_decryption = None

        # Iterate through each token, adjusting the arithmetic encoding as we
        # go. In the end, `alpha`, the low end of the bit range, will contain
        # the ciphertext encoded as an integer.
        for cumulative_prob, prev_cumulative_prob in self.model.cumprob_gen(
            covertext, self._seed, save_past=self._save_past
        ):
            bitrange: int = high - low
            low += floor(prev_cumulative_prob * bitrange)
            high -= floor((1 - cumulative_prob) * bitrange)

            shift: int = 0
            bitrange = high - low
            while bitrange <= CODING_RANGE:
                bitrange *= 2
                shift += 1

            low <<= shift
            high <<= shift
            total_encoded += shift

            if verify:
                result = self._try_decrypt(low, total_encoded, verify)
                if result is not None:
                    return result

        _logger.debug(
            "Encoded: %s"
            % (bytes_to_bits(convert_int_to_bytes(low << 8 - (total_encoded % 8))))
        )
        _logger.debug(f"Total encoded: {total_encoded}")

        result = self._try_decrypt(low, total_encoded, verify)
        if result is not None:
            return result

        _logger.error(f"Decoding failed: Unable to decrypt valid ciphertext")
        raise UnableToDecryptValidCiphertext

    @staticmethod
    def fixed_width_bitrange(low: MutableUInt32, high: MutableUInt32) -> MutableUInt32:
        assert high >= low
        return high - low + 1 if high - low != MutableUInt32.maxval else MutableUInt32(MutableUInt32.maxval)  # type: ignore

    @staticmethod
    def fixed_width_adjust(
        low: MutableUInt32,
        bitrange: MutableUInt32,
        prev_cumulative_prob: float,
        cumulative_prob: float,
    ) -> Tuple[MutableUInt32, MutableUInt32]:
        high = low + floor(float(bitrange) * cumulative_prob)  # type: ignore
        low = low + floor(float(bitrange) * prev_cumulative_prob)  # type: ignore
        return (low, high)

    def encode_fixed_width(
        self,
        ciphertext: bytes,
        complete_sentence: bool = True,
    ) -> str:
        # This approach uses the approaches defined in the video
        # `<https://www.youtube.com/watch?v=EqKbT3QdtOI&list=PLU4IQLU9e_OrY8oASHx0u3IXAL9TOdidm&index=14>`_
        # and the book Introduction to Data Compression, Chapter 4
        # `<http://students.aiu.edu/submissions/profiles/resources/onlineBook/E3B9W5_data%20compression%20computer%20information%20technology.pdf>`.

        ciphertext_bits = self.encrypter.ciphertextbits(ciphertext)

        # Last token chosen based on ciphertext bits.
        last_token: Optional[str] = None
        # Any past state from the model.
        past: Optional[Any] = None
        # The produced covertext.
        covertext: str = ""
        # List of token indices chosen. It is used to check whether the
        # generated covertext after tokenization would result in the same list
        # of token indices.
        token_indices: List[int] = []
        # List to store which cumulative probability index the algorithm picks
        # during encoding.
        cumprob_indices: List[int] = []
        # Max number of bits to encode.
        max_bits = len(ciphertext) * 8 + FIXED_POINT_EXTRA_BITS
        # Keeps track of the number of bits encoded so far.
        num_encoded_bits: int = 0
        # The low bitrange for arithmetic decoding.
        low: MutableUInt32 = MutableUInt32(MutableUInt32.minval)
        # The high bitrange for arithmetic decoding.
        high: MutableUInt32 = MutableUInt32(MutableUInt32.maxval)
        # The value we are encoding, which we use when selecting the next token.
        # We maintain the invariant that `low <= encoded <= high`.
        encoded: MutableUInt32 = MutableUInt32(ciphertext_bits.get(MutableUInt32.width))

        # Used for debugging to see whether we encode the ciphertext bits.
        if _logger.isEnabledFor(logging.DEBUG):
            to_be_taken_out: str = format(encoded, f"0{MutableUInt32.width}b")
            took_out: str = ""

        done = False
        niters: int = 0
        while not done:
            t0 = time()
            input: str
            if past is None:
                input = self._seed
            else:
                assert last_token is not None
                input = last_token if self._save_past else self._seed + covertext
            cumprobs, past = self.model.cumprob_from_input_and_past(
                input, past, use_past=self._save_past
            )

            # Find the first token index with cumulative probability greater
            # than the probability scaled by the bitrange.
            i: int = 0
            assert low <= encoded <= high
            bitrange = TextCover.fixed_width_bitrange(low, high)
            while i < len(cumprobs) and cumprobs[i][1] < (encoded - low) / bitrange:
                i += 1
            # If we exited the while loop without hitting the `and` condition,
            # set `i` to the last possible cumulative probability.
            if i == len(cumprobs):
                i = i - 1

            # Get the token associated with the given token index and add it to
            # `covertext`.
            last_token = self.model.get_token(cumprobs[i][0])
            covertext += last_token
            token_indices.append(cumprobs[i][0])
            cumprob_indices.append(i)

            (low, high) = TextCover.fixed_width_adjust(
                low,
                bitrange,
                cumprobs[i - 1][1] if i > 0 else 0,
                cumprobs[i][1],
            )

            while True:
                if low[-1] == high[-1]:
                    next_encoding_bit = ciphertext_bits.get(1)
                    num_encoded_bits += 1

                    low = low << 1  # type: ignore
                    high = (high << 1) | 1  # type: ignore
                    encoded = (encoded << 1) | next_encoding_bit  # type: ignore

                    if _logger.isEnabledFor(logging.DEBUG):
                        took_out += to_be_taken_out[0]
                        to_be_taken_out = to_be_taken_out[1:] + str(next_encoding_bit)
                elif low[-2] == 1 and high[-2] == 0:
                    next_encoding_bit = ciphertext_bits.get(1)
                    num_encoded_bits += 1

                    low[-2] = low[-1]
                    high[-2] = high[-1]
                    encoded[-2] = encoded[-1]

                    low = low << 1  # type: ignore
                    high = high << 1 | 1  # type: ignore
                    encoded = encoded << 1 | next_encoding_bit  # type: ignore

                    if _logger.isEnabledFor(logging.DEBUG):
                        took_out += to_be_taken_out[0]
                        to_be_taken_out = to_be_taken_out[1:] + str(next_encoding_bit)
                else:
                    break

                if num_encoded_bits >= max_bits:
                    if complete_sentence:
                        if covertext[-1] == ".":
                            done = True
                            break
                    else:
                        done = True
                        break

            t1 = time()
            niters += 1
            _logger.info(
                f"Iteration {niters}: {t1 - t0:.4f}s ({num_encoded_bits} / {max_bits})"
            )

        (nciphertextbits, nextrabits) = ciphertext_bits.stats()
        _logger.info(f"# ciphertext bits: {nciphertextbits}")
        _logger.info(f"# extra bits: {nextrabits}")

        _logger.debug(
            f"Ciphertext bits: {''.join(format(byte, '08b') for byte in ciphertext)}"
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(f"Encoded bits:    {took_out}")

        # Encoding fails if tokenizing the covertext doesn't produce the same
        # tokens produced during encoding.
        if self.model.tokenize(covertext) != token_indices:
            _logger.error(f"Encoding failed: Unable to tokenize produced covertext")
            _logger.debug(f"  Covertext:·{covertext}·")
            _logger.debug(f"  Tokenization: {self.model.tokenize(covertext)}")
            _logger.debug(f"  Expected:     {token_indices}")
            raise UnableToTokenizeCovertext

        _logger.info(f"Done encoding! Number of iterations: {niters}")
        _logger.debug(f"Covertext:·{covertext}·")
        return covertext

    def decode_fixed_width(
        self,
        covertext: str,
        verify: bool = False,
    ) -> Union[bool, str]:
        # The low bitrange for arithmetic encoding.
        low: MutableUInt32 = MutableUInt32(MutableUInt32.minval)
        # The high bitrange for arithmetic encoding.
        high: MutableUInt32 = MutableUInt32(MutableUInt32.maxval)
        # Reset the "finish decryption" function.
        self.finish_decryption = None
        # The (eventual) ciphertext, encoded as an integer.
        encoded: int = 0
        # The number of encoded bits.
        total_encoded: int = 0
        underflow_counter: int = 0

        for cumulative_prob, prev_cumulative_prob in self.model.cumprob_gen(
            covertext, self._seed, save_past=self._save_past
        ):
            bitrange = TextCover.fixed_width_bitrange(low, high)
            (low, high) = TextCover.fixed_width_adjust(
                low, bitrange, prev_cumulative_prob, cumulative_prob
            )

            while True:
                if low[-1] == high[-1]:
                    value = int(low[-1])
                    encoded = (encoded << 1) + value
                    total_encoded += 1

                    while underflow_counter > 0:
                        encoded = (encoded << 1) + (not value)
                        total_encoded += 1
                        underflow_counter -= 1

                    low = low << 1  # type: ignore
                    high = (high << 1) | 1  # type: ignore
                elif low[-2] == 1 and high[-2] == 0:
                    low[-2] = low[-1]
                    high[-2] = high[-1]

                    low = low << 1  # type: ignore
                    high = high << 1 | 1  # type: ignore
                    underflow_counter += 1
                else:
                    if verify:
                        result = self._try_decrypt(encoded, total_encoded, verify)
                        if result is not None:
                            return result
                    break

        _logger.debug(
            "Encoded: %s"
            % (bytes_to_bits(convert_int_to_bytes(encoded << 8 - (total_encoded % 8))))
        )
        _logger.debug(f"Total encoded: {total_encoded}")

        result = self._try_decrypt(encoded, total_encoded, verify)
        if result is not None:
            return result

        _logger.error(f"Decoding failed: Unable to decrypt valid ciphertext")
        raise UnableToDecryptValidCiphertext

    def _try_decrypt(
        self, encoded: int, total_encoded: int, verify: bool = False
    ) -> Optional[Union[bool, str]]:
        """
        Try decrypting `encoded`, returning `None` on failure, and either a bool
        if `verify` is set to `True` or the decrypted string if `verify` is set
        to `False`.

        Note! It is essential that before calling this method in a loop,
        `self.finish_decryption` is set to `None`!

        Args:
            encoded (int): The value to decrypt, encoded as an integer.

            total_encoded (int): The total number of ciphertext bits in
            `encoded`.

            verify (bool, optional): Whether to only verify decryption or not.
            Defaults to False.

        Raises:
            SentinelCheckFailed: The sentinel check failed.
            ExtraBitsNotValid: The extra bits appended to the ciphertext are not valid.

        Returns:
            Optional[Union[bool, str]]: Either a bool denoting whether
            verification succeeded (if `verify = True`) or the plaintext (if
            `verify = False`), or `None` if decryption failed.
        """
        # Only try decrypting if we have enough of the ciphertext.
        if (
            floor(total_encoded / 8)
            >= self.encrypter.bytes_to_check() + self._extra_bytes_to_check
        ):
            # Convert the integer encoding of the ciphertext to bytes and check if
            # it's valid. We use `total_encoded` to shift things such that the
            # integer we're converting falls on a byte boundary.
            ct = convert_int_to_bytes(encoded << 8 - (total_encoded % 8))

            if not self.finish_decryption:
                self.finish_decryption = self.encrypter.begin_decryption(
                    ct[: self.encrypter.bytes_to_check()]
                )
                char = "✔" if self.finish_decryption is not None else "❌"
                _logger.debug(
                    f"Bits checked: {bytes_to_bits(ct[: self.encrypter.bytes_to_check()])} {char}"
                )

            if verify:
                return self.finish_decryption is not None
            else:
                if self.finish_decryption is not None:
                    # We don't know how long the properly encrypted message is.
                    # So try decrypting all possible lengths!
                    for end in range(self.encrypter.bytes_of_nonce(), len(ct)):
                        plaintext = self.finish_decryption(
                            ct[self.encrypter.bytes_of_nonce() : end]
                        )
                        if plaintext is not None:
                            # Now, we need to check that the bonus bits are valid.
                            n = total_encoded - end * 8 - self._bonus_bits_to_skip
                            n = n if n > 0 else 0
                            if self.encrypter.check_extra_bits(ct[end:], ct[:end], n):
                                return plaintext.decode("utf-8")
                            else:
                                _logger.error(f"Decoding failed: Extra bits not valid")
                                raise ExtraBitsNotValid
                    _logger.warning(
                        f"No valid ciphertext found up to length {len(ct)} bytes"
                    )
                else:
                    _logger.error(f"Decoding failed: Sentinel check failed")
                    raise SentinelCheckFailed
        return None


def convert_int_to_bytes(g: int) -> bytes:
    """Convert an integer `g` into its byte representation."""
    return bytes(int_to_bytes_r(g))


def int_to_bytes_r(g: int) -> List[int]:
    return [*int_to_bytes_r(g >> 8), g % 256] if g > 0 else []


def bytes_to_bits(b: bytes) -> str:
    return "".join(format(byte, "08b") for byte in b)


#
# Tests
#

import unittest
import random
import string


class Test(unittest.TestCase):
    def test_tokenize_never_fails(self) -> None:
        """Ensure that `tokenword` never fails on an input."""
        from mbfte.pytorch_model_wrapper import PyTorchModelWrapper

        NITERS = 1
        encoder = TextCover(PyTorchModelWrapper, "", "")
        for i in range(NITERS):
            covertext = "".join(
                random.choice(string.ascii_lowercase) for _ in range(30)
            )
            self.assertNotEqual(
                encoder.model.cumprob_gen(covertext, encoder._seed, encoder._save_past),
                None,
            )


if __name__ == "__main__":
    unittest.main()
