from abc import ABC, abstractmethod
from hmac import compare_digest
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Tuple

from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256, SHA512, SHAKE128
from Crypto.Protocol.KDF import HKDF
from Crypto.Random import get_random_bytes

from mbfte import _logger

BLOCK_SIZE: int = 16
"""The AES block size, in bytes."""
KEY_SIZE: int = 32
"""The key size, in bytes."""


class CiphertextBits:
    """
    A wrapper around a ciphertext that emits bits of the ciphertext. When the
    ciphertext runs out of bits, the bits of an XOF applied to the key and
    ciphertext is used.
    """

    def __init__(self, key: bytes, ciphertext: bytes) -> None:
        """
        Initialize the bit generator.

        Args:
            key (bytes): The key to use.
            ciphertext (bytes): The ciphertext to use.
        """

        def ciphertextbits(
            key: bytes, ciphertext: bytes
        ) -> Generator[int, bytes, None]:
            """A generator that returns each bit of `ciphertext`. When that runs
            out the generator emits bits derived from a XOF applied to `key` and
            `ciphertext`."""
            xof = CiphertextBits._xof(key, ciphertext)
            # Yield the ciphertext bit-by-bit.
            for byte in ciphertext:
                for offset in range(8):
                    self.nciphertextbits_extracted += 1
                    yield CiphertextBits._getbit(byte, offset)
            # Yield the output of `XOF(key, ciphertext)` bit-by-bit.
            while True:
                byte = xof.read(1)[0]
                for offset in range(8):
                    self.nextrabits_extracted += 1
                    yield CiphertextBits._getbit(byte, offset)

        self.generator = iter(ciphertextbits(key, ciphertext))
        self.nciphertextbits_extracted = 0
        self.nextrabits_extracted = 0

    def stats(self) -> Tuple[int, int]:
        return (self.nciphertextbits_extracted, self.nextrabits_extracted)

    def get(self, n: int) -> int:
        """Extracts `n` bits, returning the result as an `int`."""
        val = 0
        for _ in range(n):
            val = (val << 1) + next(self.generator)
        return val

    @staticmethod
    def _getbit(byte: int, offset: int) -> int:
        return (byte >> (7 - offset)) % 2

    @staticmethod
    def _xof(key: bytes, ciphertext: bytes) -> Any:
        xof = SHAKE128.new(data=key)
        xof.update(ciphertext)
        return xof

    @staticmethod
    def check_equality(key: bytes, ciphertext: bytes, extra: bytes, n: int) -> bool:
        _logger.debug(f"Checking equality for {n} bits")
        if n == 0:
            return True
        xof = CiphertextBits._xof(key, ciphertext)
        nbits: int = 0
        # TODO: Does it matter that this is not constant time?!
        for byte in extra:
            xofbyte = xof.read(1)[0]
            _logger.debug(f"Byte: {byte:08b} XOF: {xofbyte:08b}")
            for offset in range(8):
                if CiphertextBits._getbit(byte, offset) != CiphertextBits._getbit(
                    xofbyte, offset
                ):
                    return False
                nbits += 1
                if nbits == n:
                    return True
        return False


class CryptoSystem(ABC):
    """An abstract class for defining symmetric key authenticated encryption
    schemes for use in MB-FTE. These schemes are unique in that they (1) provide
    "sentinel" bytes to efficiently check whether a ciphertext will decrypt
    correctly without needing the entire ciphertext, and (2) allow the caller to
    vary the size of the nonce to trade off security with efficiency."""

    @abstractmethod
    def __init__(
        self, key: bytes, bytes_of_sentinel: int = 2, bytes_of_nonce: int = 3
    ) -> None:
        """
        Initialize the crypto system.

        Args:
            key (bytes): The symmetric key to use.

            bytes_of_sentinel (int, optional): The number of bytes for the
            sentinel. The sentinel is used when checking whether a covertext may
            contain a hidden plaintext message. The larger the sentinel value,
            the lower the false positives (messages that one thinks can be
            decoded but actually cannot).Defaults to 2.

            bytes_of_nonce (int, optional): The number of bytes for the nonce.
            The nonce is used for security, and hence a small value has a
            detrimental impact on security. So if you expect to send a lot of
            messages, choose a large value here! Defaults to 3.
        """
        pass

    @abstractmethod
    def key(self) -> bytes:
        """The symmetric key associated with this instantiation."""
        pass

    @abstractmethod
    def bytes_to_check(self) -> int:
        """The number of bytes needed in order to check the sentinel."""
        pass

    @abstractmethod
    def bytes_of_nonce(self) -> int:
        """The number of bytes in the nonce.

        It is required that these bytes come _first_ in the produced ciphertext.
        """
        pass

    @abstractmethod
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes]:
        """Encrypt `plaintext`, outputting the nonce and ciphertext."""
        pass

    @abstractmethod
    def begin_decryption(
        self, nonce_and_sentinel: bytes
    ) -> Optional[Callable[[bytes], Optional[bytes]]]:
        """Check whether the byte array containing the nonce and (encrypted)
        sentinel is valid. If so, a function is returned which completes the
        decryption on the ciphertext."""
        pass

    def ciphertextbits(self, ciphertext: bytes) -> CiphertextBits:
        """
        Return a `CiphertextBits` object built from `ciphertext`.

        Args:
            ciphertext (bytes): The ciphertext to use.

        Returns:
            CiphertextBits: A bit generator for `ciphertext`.
        """
        return CiphertextBits(self.key(), ciphertext)

    def check_extra_bits(self, extras: bytes, ciphertext: bytes, n: int) -> bool:
        """
        Check that the `n` extra bits in `extras` are valid.

        Args:
            extras (bytes): The extra bits to check.
            ciphertext (bytes): The original ciphertext.
            n (int): The number of bits of `extras` to check.

        Returns:
            bool: `True` if the bits are valid, `False` otherwise.
        """
        return CiphertextBits.check_equality(self.key(), ciphertext, extras, n)


class RandomPadding(CryptoSystem):
    """This class prepends bytes of random padding to the `SivWithPadding` crypto system."""

    def __init__(
        self,
        key: bytes,
        bytes_of_padding: int = 3,
        bytes_of_sentinel: int = 2,
        bytes_of_nonce: int = 3,
    ):
        self._crypto = SivWithPadding(key, bytes_of_sentinel, bytes_of_nonce)
        self._bytes_of_padding = bytes_of_padding

    def key(self) -> bytes:
        return self._crypto.key()

    def bytes_to_check(self) -> int:
        return self._crypto.bytes_to_check()

    def bytes_of_nonce(self) -> int:
        return self._crypto.bytes_of_nonce()

    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes]:
        padding = get_random_bytes(self._bytes_of_padding)
        return self._crypto.encrypt(padding + plaintext)

    def begin_decryption(
        self, nonce_and_sentinel: bytes
    ) -> Optional[Callable[[bytes], Optional[bytes]]]:
        finish = self._crypto.begin_decryption(nonce_and_sentinel)
        if finish is None:
            return None
        else:

            def finish_decryption(ciphertext: bytes) -> Optional[bytes]:
                plaintext = finish(ciphertext)
                if plaintext is None:
                    return None
                else:
                    return plaintext[self._bytes_of_padding :]

            return finish_decryption


class SivWithPadding(CryptoSystem):
    """This class defines a deterministic symmetric key encryption scheme as follows::

        Encrypt(K1, K2, SV, M):
            N <-- PRF_{K1}(SV || M)
            C <-- CTR[AES_{K2}](N, SV || M)
            Return (N, C)

        Decrypt(K1, K2, N, C):
            SV, M <-- CTR[AES_{K2}](N, C)
            If SV is invalid then abort
            If N != PRF_{K1}(SV || M) then abort
            Return M

    Here, `N` denotes the nonce, `SV` denotes the sentinel, and the keys `K1` and `K2`
    are derived from some input key `K`. (Note that in this implementation, `SV` is
    set to all zeros.)"""

    def __init__(
        self,
        key: bytes,
        bytes_of_sentinel: int = 2,
        bytes_of_nonce: int = 3,
    ) -> None:
        """
        Initializes the crypto system.

        Raises:
         ValueError: If `bytes_of_sentinel` or `bytes_of_nonce` are larger than `BLOCK_SIZE`.
        """
        assert len(key) == KEY_SIZE, format(
            f"key length invalid: got {len(key)} bytes, need 32"
        )  # nosec
        keys = HKDF(
            key,
            32,
            salt=b"",
            context=b"May 6th 2022 butkus cryptography",
            num_keys=2,
            hashmod=SHA512,
        )
        if TYPE_CHECKING:
            assert isinstance(keys, tuple)  # nosec
        nonce_key, encryption_key = keys
        self._nonce_key = nonce_key
        self._encryption_key = encryption_key
        self._key = key
        if bytes_of_sentinel > BLOCK_SIZE:
            raise ValueError(
                f"`bytes_of_sentinel` cannot be larger than the block size ({BLOCK_SIZE})"
            )
        self.bytes_of_sentinel = bytes_of_sentinel
        if bytes_of_nonce > BLOCK_SIZE:
            raise ValueError(
                f"`bytes_of_nonce` cannot be larger than the block size ({BLOCK_SIZE})"
            )
        self._bytes_of_nonce = bytes_of_nonce
        self._bytes_to_check = bytes_of_sentinel + bytes_of_nonce

    def key(self) -> bytes:
        return self._key

    def bytes_to_check(self) -> int:
        return self._bytes_to_check

    def bytes_of_nonce(self) -> int:
        return self._bytes_of_nonce

    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt `plaintext`, outputting the nonce and ciphertext.

        This method prepends a number of random bytes to the plaintext to
        provide randomness in the encryption, where the number of random bytes
        is specified during object instantiation.
        """
        sentinel = bytearray(self.bytes_of_sentinel)
        expanded_plaintext = sentinel + plaintext
        # Compute `N <-- PRF_{K1}(SV || M)`
        hmac = HMAC.new(self._nonce_key, digestmod=SHA256)
        hmac.update(expanded_plaintext)
        nonce = hmac.digest()[0 : self._bytes_of_nonce]
        # Compute `C <-- CTR[AES_{K2}(N, SV || M)]`
        aes = AES.new(self._encryption_key, AES.MODE_ECB)
        # Manually implement CTR mode, since we need to control the nonce.
        # TODO: We probably want some proper padding here.
        ciphertext = b""
        for i, block in enumerate(
            [
                expanded_plaintext[i : i + 16]
                for i in range(0, len(expanded_plaintext), 16)
            ]
        ):
            nonce_ = self._full_nonce(nonce, i)
            block_ = aes.encrypt(nonce_)
            ciphertext_block = SivWithPadding._xor(block, block_)
            ciphertext += ciphertext_block
        return (nonce, ciphertext)

    def _full_nonce(self, nonce: bytes, incr: int) -> bytes:
        assert len(nonce) == self._bytes_of_nonce
        padded_nonce = nonce + (b"\0" * (BLOCK_SIZE - self._bytes_of_nonce))
        return (int.from_bytes(padded_nonce, "big") + incr).to_bytes(BLOCK_SIZE, "big")

    @staticmethod
    def _xor(block: bytes, block_: bytes) -> bytes:
        return bytes(a ^ b for a, b in zip(block, block_))

    def begin_decryption(
        self, nonce_and_sentinel: bytes
    ) -> Optional[Callable[[bytes], Optional[bytes]]]:
        """
        Check whether the byte array containing the nonce and (encrypted)
        sentinel is valid. If so, a function is returned which completes the
        decryption on the ciphertext.
        """
        assert len(nonce_and_sentinel) == self._bytes_to_check
        # Extract the nonce and (encrypted) sentinel from the provided bytes.
        nonce = nonce_and_sentinel[: self._bytes_of_nonce]
        assert len(nonce) == self._bytes_of_nonce
        sentinel = nonce_and_sentinel[self._bytes_of_nonce : self._bytes_to_check]
        assert len(sentinel) == self.bytes_of_sentinel
        # Compute `SV <-- CTR[AES_{K2}](N, encrypted-SV)`
        aes = AES.new(self._encryption_key, AES.MODE_ECB)
        # Run CTR mode on the encrypted sentinel. We pad the encrypted sentinel
        # block with zeros because we're only checking here that the sentinel is
        # valid.
        nonce_ = self._full_nonce(nonce, 0)
        block_ = aes.encrypt(nonce_)
        sentinel_ = sentinel + (b"\0" * (BLOCK_SIZE - self.bytes_of_sentinel))
        block_ = SivWithPadding._xor(block_, sentinel_)
        # Check that `SV` is indeed all zeros.
        if not compare_digest(
            b"\x00" * self.bytes_of_sentinel,
            block_[0 : self.bytes_of_sentinel],  # noqa: E203
        ):
            return None

        def finish_decryption(ciphertext: bytes) -> Optional[bytes]:
            """
            Complete the decryption. `ciphertext` should include the encrypted
            sentinel, but _not_ the nonce.
            """
            # Compute `SV, M <-- CTR[AES_{K2}](N, C)`
            plaintext = b""
            for i, block in enumerate(
                [
                    ciphertext[i : i + BLOCK_SIZE]
                    for i in range(0, len(ciphertext), BLOCK_SIZE)
                ]
            ):
                nonce_ = self._full_nonce(nonce, i)
                block_ = aes.encrypt(nonce_)
                plaintext += SivWithPadding._xor(block, block_)
            # Check that the nonce is correct.
            hmac = HMAC.new(self._nonce_key, digestmod=SHA256)
            hmac.update(plaintext)
            recomputed_nonce = bytearray(hmac.digest()[0 : self._bytes_of_nonce])
            if not compare_digest(nonce, recomputed_nonce):
                return None
            # Strip off the sentinel and padding when returning the plaintext.
            return plaintext[self.bytes_of_sentinel :]

        return finish_decryption


#
# Tests
#

import unittest


class Test(unittest.TestCase):
    def test_siv_with_padding_works(self) -> None:
        """Ensure that `SivWithPadding` works as expected."""
        import os

        for _ in range(100):
            key = os.urandom(32)
            cs = SivWithPadding(key)

            def test_roundtrip(msg: bytes) -> None:
                nonce, ciphertext = cs.encrypt(msg)
                finish = cs.begin_decryption(nonce + ciphertext[: cs.bytes_of_sentinel])
                self.assertNotEqual(finish, None)
                assert finish is not None  # Needed to make `mypy` happy
                self.assertEqual(finish(ciphertext), msg)

            test_roundtrip(b"")
            for n in [1, 5, 86, 938, 1024, 8192]:
                for _ in range(20):
                    test_roundtrip(os.urandom(n))


if __name__ == "__main__":
    unittest.main()
