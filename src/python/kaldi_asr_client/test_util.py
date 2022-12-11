import random
import sys

# LibriSpeech dataset
DATA_DIR = "/data"
DATASET = "LibriSpeech"
WAV_BASEDIR = f"{DATA_DIR}/data/{DATASET}/test-clean/"
SAMP_FREQ = 16000


def get_wavs(wav_id):
    basedir = f"{WAV_BASEDIR}/{wav_id}"

    # (bytes, expected inference)
    wavs_out = []

    with open(f"{basedir}/{wav_id.replace('/', '-')}.trans.txt", "r") as f:
        for line in f.readlines():
            line = line.split(" ")

            wav = line[0]
            expected = (
                " ".join(line[1:]).replace("\n", "") + " "
            )  # Trailing space necessary

            with open(f"{basedir}/{wav}.wav", "rb") as wf:
                wavs_out.append((wav, wf.read(), expected))

    print(f"Loaded dataset {wav_id} with {len(wavs_out)} WAVs for testing")
    return wavs_out


def samp_freq():
    return SAMP_FREQ


def test_infer(client):
    wavs = []

    if len(sys.argv) < 2:
        raise Exception("No WAV directory specified!")

    # 908/157963
    for wav_id in sys.argv[1:]:
        wavs.extend(get_wavs(wav_id))

    random.shuffle(wavs)

    for idx, inferred in enumerate(
        client.infer(list(map(lambda item: item[1], wavs)))
    ):
        if wavs[idx][2] != inferred:
            print(
                f"Output for {wavs[idx][0]} does not match!\nResult: {inferred}\nExpected: {wavs[idx][2]}\n"
            )
