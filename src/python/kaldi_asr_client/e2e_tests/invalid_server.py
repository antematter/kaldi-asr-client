import sys

import test_util
from kaldi_asr_client import Client

try:
    with Client(
        samp_freq=50,
        servers=["localhost:6666"],
    ) as client:
        pass
except Exception as e:
    print(f"Caught exception: {e}")
    sys.exit(0)

raise Exception("FAILED")
