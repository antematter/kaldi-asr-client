import test_util
from kaldi_asr_client import Client

with Client(
    samp_freq=test_util.samp_freq(),
    servers=["localhost:8001", "localhost:8002"],
) as client:
    test_util.test_infer(client)
