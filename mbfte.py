"""
A program for running MB-FTE.
"""

import click
from json import dumps
from statistics import mean, median, stdev
import string
from time import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Type, Union
import numpy as np
from mbfte.textcover import (
    ExtraBitsNotValid,
    SentinelCheckFailed,
    TextCover,
    TooManyZeroShifts,
    UnableToDecryptValidCiphertext,
    UnableToTokenizeCovertext,
)
from mbfte.model_wrapper import AbstractModelWrapper
import logging
import random
import sys


_logger = logging.getLogger(__name__)


def bool_to_symbol(b: bool) -> str:
    return "✔" if b else "❌"


def random_string(num: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits + " ") for _ in range(num)
    )


def roundtrip(
    plaintext: str,
    key: bytes,
    model: TextCover,
    model2: TextCover,
    complete_sentence: bool = True,
) -> Dict[str, Any]:
    """Test the full encode-decode cycle and collect any relevant metrics.

    Args:
     plaintext:
        The plaintext string to encode.
     keys:
        A list of keys to use. The first key provided will be used to encode the
        message, and the other keys are used for checking validity of a message
        against the wrong keys.
     model:
        The "good" model to use when encoding / decoding.
     model2:
        The "bad" model to use to validate the decoding fails when using the
        wrong model.

    Returns:
     A dictionary containing various relevant metrics.
    """
    _logger.info("Encoding plaintext")
    start = time()
    covertext = model.encode(plaintext, complete_sentence)
    end = time()
    encode_time = end - start

    _logger.info("Checking covertext")
    start = time()
    ok = model.check(covertext)
    end = time()
    good_check_time = end - start
    if not ok:
        _logger.error("Decoding check failed.")
        _logger.error(f"  Plaintext:·{plaintext}·")
        _logger.error(f"  Covertext:·{covertext}·")
        _logger.error(f"  Key:·{key.hex()}·")
        assert False, "Decoding check should never fail"

    _logger.info("Decoding covertext")
    start = time()
    plaintext_ = model.decode(covertext)
    end = time()
    decode_time = end - start

    _logger.info("Decoding covertext with wrong model")
    start = time()
    ok2 = model2.check(covertext)
    end = time()
    bad_check_time = end - start
    # Check using `model2` should fail, since the seed is different.
    if ok2:
        _logger.error("Decoding check succeeded on wrong model.")
        _logger.error(f"  Plaintext:·{plaintext}·")
        _logger.error(f"  Covertext:·{covertext}·")
        _logger.error(f"  Key:·{key.hex()}·")
        assert False, "Decoding check on wrong model should never succeed"

    if plaintext_ != plaintext:
        _logger.error("Decoding check succeeded but decoding failed.")
        _logger.error(f"  Plaintext:·{plaintext}·")
        _logger.error(f"  Covertext:·{covertext}·")
        _logger.error(f"  Decoded plaintext:·{plaintext_}·")
        _logger.error(f"  Key:·{key.hex()}·")
        assert False, "Decoding should never fail"

    _logger.info("Tokenizing covertext")
    token_indices = model.model.tokenize(covertext)
    _logger.info("Computing cumulative probability indices")
    cumprob_indices = model.model.cumprob_indices(covertext, model.seed())

    return {  # all times in seconds
        "encode time": encode_time,
        "good check time": good_check_time,
        "decode time": decode_time,
        "bad check time": bad_check_time,
        "plaintext": plaintext,
        "covertext": covertext,
        "expansion": len(covertext) / len(plaintext),
        "key": key.hex(),
        "tokens": token_indices,
        "cumprobs": cumprob_indices,
    }


def benchmark(
    length: int,
    n_trials: int,
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    temperature: float = 0.8,
    padding: int = 3,
    complete_sentence: bool = True,
    save_past: bool = True,
    arithmetic_encoding_bitwidth: str = "variable",
) -> Dict[str, Any]:
    """
    Collect MB-FTE statistics.
    """

    # `random.randbytes` only available in versions >= 3.9.
    # key = random.randbytes(32)
    key = random.getrandbits(256).to_bytes(length=32, byteorder="little")

    model = TextCover(
        model_wrapper,
        model_dir,
        seed,
        key,
        temperature,
        padding,
        save_past,
        arithmetic_encoding_bitwidth,
    )

    model2 = TextCover(
        model_wrapper,
        model_dir,
        # Make the seed different, so the model is different.
        seed + " ",
        key,
        temperature,
        padding,
        save_past,
        arithmetic_encoding_bitwidth,
    )

    trials = []
    covertexts: List[str] = []
    plaintexts: List[str] = []
    tokens: List[List[int]] = []
    cumprobs: List[List[int]] = []
    failure_too_many_zero_shifts: int = 0
    failure_unable_to_tokenize_covertext: int = 0
    failure_sentinel_check_failed: int = 0
    failure_unable_to_decrypt: int = 0
    _logger.info(f"Key: {key.hex()}")
    for _ in range(n_trials):
        plaintext = random_string(length)
        _logger.info(f"Plaintext:·{plaintext}·")
        try:
            t = roundtrip(plaintext, key, model, model2, complete_sentence)
            trials.append(t)
            _logger.info(f"Result: {dumps(t)}")
            covertexts.append(t["covertext"])
            plaintexts.append(t["plaintext"])
            tokens.append(t["tokens"])
            cumprobs.append(t["cumprobs"])
        except TooManyZeroShifts:
            failure_too_many_zero_shifts += 1
        except UnableToTokenizeCovertext:
            failure_unable_to_tokenize_covertext += 1
        except SentinelCheckFailed:
            failure_sentinel_check_failed += 1
        except UnableToDecryptValidCiphertext:
            assert False, "We should always be able to decrypt a valid ciphertext."
        except ExtraBitsNotValid:
            assert False, "Extra bits should always be good."
    # Extract non-`None` data from `trials`.
    f: Callable[[str], List[float]] = lambda name: [
        float(i)  # type: ignore
        for i in filter(lambda x: x is not None, (t.get(name) for t in trials))
    ]
    encode_time = f("encode time")
    good_check_time = f("good check time")
    decode_time = f("decode time")
    bad_check_time = f("bad check time")
    expansion = f("expansion")
    # Collect relevant statistics.
    compute_mean = lambda values: mean(values) if len(values) else float("nan")
    compute_stdev = lambda values: stdev(values) if len(values) > 1 else float("nan")
    compute_median = lambda values: median(values) if len(values) else float("nan")
    successful = (
        n_trials
        - (failure_too_many_zero_shifts + failure_unable_to_tokenize_covertext)
        - (failure_sentinel_check_failed + failure_unable_to_decrypt)
    )
    result: Dict[str, Any] = {
        "model": {
            "type": model_wrapper.NAME,
            "dir": model_dir,
        },
        "plaintext length": length,
        "trials": {
            "total": n_trials,
            "successful": successful,
            "rate": f"{int((successful / n_trials) * 100)}%",
        },
        "encode failures": {
            "total": failure_too_many_zero_shifts
            + failure_unable_to_tokenize_covertext,
            "too many zero shifts": failure_too_many_zero_shifts,
            "unable to tokenize covertext": failure_unable_to_tokenize_covertext,
        },
        "decode failures": {
            "total": failure_sentinel_check_failed + failure_unable_to_decrypt,
            "sentinel check failed": failure_sentinel_check_failed,
            "unable to decrypt": failure_unable_to_decrypt,
        },
        "encode": {
            "mean": f"{compute_mean(encode_time):.4f} +- {compute_stdev(encode_time):.4f}",
            "median": f"{compute_median(encode_time):.4f}",
        },
        "decode": {
            "mean": f"{compute_mean(decode_time):.4f} +- {compute_stdev(decode_time):.4f}",
            "median": f"{compute_median(decode_time):.4f}",
        },
        "good check": {
            "mean": f"{compute_mean(good_check_time):.4f} +- {compute_stdev(good_check_time):.4f}",
            "median": f"{compute_median(good_check_time):.4f}",
        },
        "bad check": {
            "mean": f"{compute_mean(bad_check_time):.4f} +- {compute_stdev(bad_check_time):.4f}",
            "median": f"{compute_median(bad_check_time):.4f}",
        },
        "expansion": {
            "mean": f"{compute_mean(expansion):.4f} +- {compute_stdev(expansion):.4f}",
            "median": f"{compute_median(expansion):.4f}",
        },
    }
    click.echo("Stats:")
    click.echo(dumps(result, indent=4))
    click.echo()
    for i, (plaintext, covertext, tokens_, cumprobs_) in enumerate(
        zip(plaintexts, covertexts, tokens, cumprobs), start=1
    ):
        click.echo(f"#{i}·{plaintext}·")
        click.echo(f"#{i}·{covertext}·")
        click.echo(f"#{i} Token indices:    {tokens_}")
        click.echo(f"#{i} Cum prob indices: {cumprobs_}")
    return result


def encode(
    plaintext: str,
    key: bytes,
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    temperature: float = 0.8,
    padding: int = 3,
    complete_sentence: bool = True,
    save_past: bool = True,
    arithmetic_encoding_bitwidth: str = "variable",
) -> str:
    """Run the encode operation on a plaintext."""
    model = TextCover(
        model_wrapper,
        model_dir,
        seed,
        key,
        temperature,
        padding,
        save_past,
        arithmetic_encoding_bitwidth,
    )

    covertext = model.encode(plaintext, complete_sentence)
    click.echo(f"Produced covertext:·{covertext}·")
    return covertext


def decode(
    covertext: str,
    key: bytes,
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    temperature: float = 0.8,
    padding: int = 3,
    verify: bool = False,
    save_past: bool = True,
    arithmetic_encoding_bitwidth: str = "variable",
) -> Union[bool, str]:
    """Run the decode operation on a covertext."""
    model = TextCover(
        model_wrapper,
        model_dir,
        seed,
        key,
        temperature,
        padding,
        save_past,
        arithmetic_encoding_bitwidth,
    )
    if verify:
        ok = model.check(covertext)
        if ok:
            click.echo("Verification succeeded 👍")
        else:
            click.echo("Verification failed ❌")
        return ok
    else:
        tokens = model.model.tokenize(covertext)
        cumprobs = model.model.cumprob_indices(covertext, seed, save_past)
        click.echo(f"Token indices:    {tokens}")
        click.echo(f"Cum prob indices: {cumprobs}")
        plaintext = model.decode(covertext)
        click.echo(f"Produced plaintext:·{plaintext}·")
        return plaintext


def encode_decode(
    plaintext: str,
    key: bytes,
    seed: str,
    model_dir: str,
    model_wrapper: Type[AbstractModelWrapper],
    temperature: float = 0.8,
    padding: int = 3,
    verify: bool = False,
    complete_sentence: bool = False,
    save_past: bool = True,
    arithmetic_encoding_bitwidth: str = "variable",
) -> None:
    """For testing a one-off run (for example, if encoding fails and we want to
    try to diagnose why).

    Args:
     plaintext:
        The plaintext we want to encode and decode.
     key:
        The key, given in hex.
     seed:
        The model seed.
    """
    covertext = encode(
        plaintext,
        key,
        seed,
        model_dir,
        model_wrapper,
        temperature,
        padding,
        complete_sentence,
        save_past,
        arithmetic_encoding_bitwidth,
    )
    result = decode(
        covertext,
        key,
        seed,
        model_dir,
        model_wrapper,
        temperature,
        padding,
        verify,
        save_past,
        arithmetic_encoding_bitwidth,
    )
    if TYPE_CHECKING:
        assert isinstance(result, str)
    if not verify:
        assert plaintext == result
        click.echo("Covertext decoded correctly 👍")


class LogLevel(click.ParamType):
    name = "loglevel"

    def convert(self, value, param, ctx):  # type: ignore
        try:
            return getattr(logging, value)
        except AttributeError:
            self.fail(
                f"Must be one of: DEBUG, INFO, WARN, ERROR",
                param,
                ctx,
            )


class Key(click.ParamType):
    name = "key"

    def convert(self, value, param, ctx) -> bytes:  # type: ignore
        try:
            key = bytes.fromhex(value)
            if len(key) != 32:
                self.fail(f"Must be hex string of length 32.", param, ctx)
            return key
        except ValueError:
            self.fail(f"Must be hex string of length 32.", param, ctx)


class ModelWrapper(click.ParamType):
    name = "modelwrapper"

    def convert(self, value, param, ctx) -> Type[AbstractModelWrapper]:  # type: ignore
        if value == "pytorch":
            from mbfte.pytorch_model_wrapper import PyTorchModelWrapper

            return PyTorchModelWrapper
        else:
            self.fail(f"Must be one of: pytorch", param, ctx)


class ArithmeticEncoding(click.ParamType):
    name = "arithmetic_encoding_bitwidth_type"

    def convert(self, value: str, param, ctx) -> str:  # type: ignore
        if value in ("variable", "fixed"):
            return value
        else:
            self.fail(f"Must be one of: variable, fixed", param, ctx)


@click.group()
@click.option(
    "--loglevel",
    metavar="LEVEL",
    help="The log level ('DEBUG', 'INFO', 'WARN', or 'ERROR').",
    type=LogLevel(),
    default="WARN",
    show_default=True,
)
@click.option(
    "--temperature",
    metavar="T",
    help="The model temperature.",
    type=float,
    default=0.8,
    show_default=True,
)
@click.option(
    "--padding",
    metavar="N",
    help="Bytes of random padding to use when encrypting.",
    type=int,
    default=3,
    show_default=True,
)
@click.option(
    "--model-type",
    metavar="TYPE",
    help="The model type (only 'pytorch' supported for now).",
    type=ModelWrapper(),
    default="pytorch",
    show_default=True,
)
@click.option(
    "--model-dir",
    metavar="DIR",
    help='The model directory, or the special text "PRETRAINED" to use a pre-trained PyTorch model.',
    type=str,
    default="PRETRAINED",
    show_default=True,
)
@click.option(
    "--seed-text",
    metavar="TEXT",
    help="Initial seed text to use in the model (generally longer is better).",
    type=str,
    default="Here is the news of the day. ",
    show_default=True,
)
@click.option(
    "--seed-randomness",
    metavar="N",
    help="Set the randomness seed to N (to be fully deterministic you also need to set `--padding 0`).",
    type=int,
    default=None,
)
@click.option(
    "--complete-sentence/--do-not-complete-sentence",
    default=False,
    help="Whether to complete sentences when generating covertext.",
    show_default=True,
)
@click.option(
    "--save-past/--do-not-save-past",
    default=True,
    help="Whether to save past state when running model prediction (only used for 'pytorch').",
    show_default=True,
)
@click.option(
    "--arithmetic-encoding-bitwidth",
    metavar="ENCODING",
    help="Whether to use fixed or variable point bitwidth for arithmetic encoding (fixed/variable)",
    type=ArithmeticEncoding(),
    default="fixed",
    show_default=True,
)
@click.pass_context
def cli(  # type: ignore
    ctx,
    loglevel,
    temperature,
    padding,
    model_type,
    model_dir,
    seed_text,
    seed_randomness,
    complete_sentence,
    save_past,
    arithmetic_encoding_bitwidth,
) -> None:
    """Implementation of model-based format transforming encryption (MB-FTE).

    FTE is a technique for transforming a plaintext into a ciphertext such that
    the ciphertext conforms to a particular format. In MB-FTE, that format is
    the output of a large language model.
    """
    # Set up logging.
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    logging.getLogger("mbfte").level = loglevel
    logging.getLogger(__name__).level = loglevel
    # Set up randomness.
    if seed_randomness is None:
        seed_randomness = random.getrandbits(32)
    random.seed(seed_randomness)
    np.random.seed(seed_randomness)

    model_wrapper: Type[AbstractModelWrapper]
    if model_type.NAME == "pytorch":
        from mbfte.pytorch_model_wrapper import PyTorchModelWrapper
        import torch

        torch.manual_seed(seed_randomness)
        model_wrapper = PyTorchModelWrapper
    else:
        assert False, "Uknown model type, we shouldn't get here!"

    ctx.ensure_object(dict)
    ctx.obj["seed-randomness"] = seed_randomness
    ctx.obj["temperature"] = temperature
    ctx.obj["padding"] = padding
    ctx.obj["model-wrapper"] = model_wrapper
    ctx.obj["model-dir"] = model_dir
    ctx.obj["seed-text"] = seed_text
    ctx.obj["complete-sentence"] = complete_sentence
    ctx.obj["save-past"] = save_past
    ctx.obj["arithmetic-encoding-bitwidth"] = arithmetic_encoding_bitwidth


def log_user_settings(ctx: click.Context) -> None:
    _logger.info(f"Randomness seed: {ctx.obj['seed-randomness']}")
    _logger.info(f"Temperature: {ctx.obj['temperature']}")
    _logger.info(f"Padding: {ctx.obj['padding']}")
    _logger.info(f"Seed text:·{ctx.obj['seed-text']}·")
    _logger.info(f"Complete sentences? {bool_to_symbol(ctx.obj['complete-sentence'])}")
    _logger.info(f"Save past? {bool_to_symbol(ctx.obj['save-past'])}")
    _logger.info(
        f"Arithmetic Encoding Bitwidth: {ctx.obj['arithmetic-encoding-bitwidth']}"
    )


@cli.command("encode")
@click.argument("plaintext", type=str)
@click.argument("key", type=Key())
@click.pass_context
def cmd_encode(ctx: click.Context, plaintext: str, key: bytes) -> None:
    """Run encode on PLAINTEXT and KEY."""
    log_user_settings(ctx)
    try:
        encode(
            plaintext,
            key,
            ctx.obj["seed-text"],
            ctx.obj["model-dir"],
            ctx.obj["model-wrapper"],
            temperature=ctx.obj["temperature"],
            padding=ctx.obj["padding"],
            complete_sentence=ctx.obj["complete-sentence"],
            save_past=ctx.obj["save-past"],
            arithmetic_encoding_bitwidth=ctx.obj["arithmetic-encoding-bitwidth"],
        )
    except (UnableToTokenizeCovertext, TooManyZeroShifts):
        sys.exit(1)


@cli.command("decode")
@click.argument("covertext", type=str)
@click.argument("key", type=Key())
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Whether to just verify a covertext decodes successfully.",
)
@click.pass_context
def cmd_decode(ctx: click.Context, covertext: str, key: bytes, verify: bool) -> None:
    """Run decode on COVERTEXT and KEY."""
    log_user_settings(ctx)
    _logger.info(f"Verify only? {bool_to_symbol(verify)}")
    try:
        decode(
            covertext,
            key,
            ctx.obj["seed-text"],
            ctx.obj["model-dir"],
            ctx.obj["model-wrapper"],
            temperature=ctx.obj["temperature"],
            padding=ctx.obj["padding"],
            save_past=ctx.obj["save-past"],
            verify=verify,
            arithmetic_encoding_bitwidth=ctx.obj["arithmetic-encoding-bitwidth"],
        )
    except (SentinelCheckFailed, UnableToDecryptValidCiphertext):
        sys.exit(1)


@cli.command("encode-decode")
@click.argument("plaintext", type=str)
@click.argument("key", type=Key())
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Whether to just verify a covertext decodes successfully.",
)
@click.pass_context
def cmd_encode_decode(
    ctx: click.Context, plaintext: str, key: bytes, verify: bool
) -> None:
    """Run encode and decode on PLAINTEXT and KEY."""
    log_user_settings(ctx)
    _logger.info(f"Verify only? {bool_to_symbol(verify)}")
    try:
        encode_decode(
            plaintext,
            key,
            ctx.obj["seed-text"],
            ctx.obj["model-dir"],
            ctx.obj["model-wrapper"],
            temperature=ctx.obj["temperature"],
            padding=ctx.obj["padding"],
            complete_sentence=ctx.obj["complete-sentence"],
            save_past=ctx.obj["save-past"],
            verify=verify,
            arithmetic_encoding_bitwidth=ctx.obj["arithmetic-encoding-bitwidth"],
        )
    except (
        UnableToTokenizeCovertext,
        TooManyZeroShifts,
        SentinelCheckFailed,
        UnableToDecryptValidCiphertext,
    ):
        sys.exit(1)


@cli.command("benchmark")
@click.option(
    "--length",
    metavar="N",
    help="Length of each plaintext message.",
    type=int,
    default=10,
    show_default=True,
)
@click.option(
    "--ntrials",
    metavar="N",
    help="Number of trials to run.",
    type=int,
    default=10,
    show_default=True,
)
@click.pass_context
def cmd_benchmark(ctx: click.Context, length: int, ntrials: int) -> None:
    """Run benchmarks.

    This command collects statistics across several runs of encode and decode
    and outputs them to STDOUT.
    """
    log_user_settings(ctx)
    benchmark(
        length,
        ntrials,
        ctx.obj["seed-text"],
        ctx.obj["model-dir"],
        ctx.obj["model-wrapper"],
        temperature=ctx.obj["temperature"],
        padding=ctx.obj["padding"],
        complete_sentence=ctx.obj["complete-sentence"],
        save_past=ctx.obj["save-past"],
        arithmetic_encoding_bitwidth=ctx.obj["arithmetic-encoding-bitwidth"],
    )


if __name__ == "__main__":
    try:
        cli()
    except FileNotFoundError as e:
        _logger.critical(e)
        sys.exit(1)
