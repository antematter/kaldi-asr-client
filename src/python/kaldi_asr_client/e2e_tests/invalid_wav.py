import sys

from kaldi_asr_client import Client

try:
    with Client(
        samp_freq=50,
    ) as client:
        client.infer([b"invalid", b"invalid2"])
except Exception as e:
    print(f"Caught exception: {e}")
    sys.exit(0)

raise Exception("FAILED")
