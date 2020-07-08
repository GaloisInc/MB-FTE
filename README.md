# Model-based Format-transforming Encryption (MB-FTE)

An implementation of model-based format-transforming encryption as introduced in
[this paper](https://arxiv.org/abs/2110.07009). This implementation provides a
library and command-line interface for using MB-FTE to encode / decode messages.

Note: This implementation does not exactly mimic the algorithm presented in the
aforementioned paper! In particular, there are two key differences:

- We use a deterministic(-ish) symmetric key scheme as opposed to the
  nonce-based symmetric key scheme used in the paper. This it to avoid needing
  to assume reliable message delivery (so that the receiver knows the correct
  nonce to use when decrypting). See `mbfte/crypto.py` for details.
- We provide two arithmetic encoding algorithms, both of which differ from the
  one presented in the paper. The first approach encodes into a variable-width
  integer, and the second approach encodes into a fixed-width integer. See
  `mbfte/mbfte.py` for details.

## Setup

We strongly encourage using a virtual environment to isolate dependencies. It
should work with Python 3.8 or newer.

To use **for the first time**, create a new venv named "venv":

    > python -m venv venv

Then activate it:

    > source venv/bin/activate

Then install the packages that we depend on:

    (venv) > pip install -r deps.txt

Use the `deactivate` command to exit the venv.

To **reuse the venv** in a fresh shell, after it has been created and deps
installed, just run the `activate` command above.

## Running

You can run the code using the `mbfte.py` command-line program. For example,
run the following to print useful help information.

    > python mbfte.py --help

In order to run the script on anything useful, you'll need a `pytorch` model.
Examples of these can be found in the `butkuscoremodels` repo. Alternatively,
for `pytorch` you can use the default pretrained model by passing `--model-dir
PRETRAINED`.

### Running on an ARM64 docker container

To run on the provided ARM64 docker container, you'll need to install `qemu`,
and in particular, the following three packages: `qemu`, `binfmt-support`,
`qemu-user-static`.

Once installed, run the following `docker` command to set up `qemu`:

    > docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

Lastly, you can build the ARM64 docker container by running the following:

    > docker build --file ./Dockerfile_arm64 -t "arm64v8/ubuntu:butkus" .

This will build all the necessary dependencies and set up the container to run
the code.

To enter get a bash shell in the built container, run the following:

    > docker run -it arm64v8/ubuntu:butkus /bin/bash

## Documentation

You can build and view available documentation as follows:

    > cd docs
    > make html
    > <your-browser-of-choice> _build/html/index.html

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## Authors

This implementation was developed as a joint collaboration between University of
Florida and Galois, Inc. The following people have contributed to the
development:

- Luke Bauer
- Himanshu Goyal
- Alex Grushin
- Chris Phifer
- Alex J Malozemoff
- Hari Menon

## Acknowledgments

This material is based upon work supported by the Defense Advanced Research
Projects Agency (DARPA) under Contract Number FA8750-19-C-0085. Any opinions,
findings and conclusions or recommendations expressed in this material are those
of the author(s) and do not necessarily reflect the views of the DARPA.

Distribution Statement "A" (Approved for Public Release, Distribution
Unlimited)

Copyright © 2019-2023 Galois, Inc.
