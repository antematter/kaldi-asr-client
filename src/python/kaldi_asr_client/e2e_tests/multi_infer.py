import test_util
from kaldi_asr_client import Client

with Client(
    samp_freq=test_util.samp_freq(),
) as client:
    test_util.test_infer(client)
    test_util.test_infer(client)
