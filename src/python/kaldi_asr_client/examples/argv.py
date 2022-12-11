import sys

from kaldi_asr_client import Client

wavs = []

for wav in sys.argv[1:]:
    with open(wav, "rb") as wav_fd:
        wavs.append((wav, wav_fd.read()))

with Client(
    samp_freq=22050,
    servers=["localhost:8001", "localhost:8002"],
    model_name="kaldi_online",
    ncontextes=10,
    chunk_length=8160,
    verbose=False,
) as client:
    # Client.infer expects a list of WAV bytes objects
    wav_bytes = list(map(lambda item: item[1], wavs))
    for index, inference in enumerate(client.infer(wav_bytes)):
        wav_file = wavs[index][0]
        print(f"Inference for {wav_file}: {inference}")
