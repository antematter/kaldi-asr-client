# kaldi-asr-client

# Building

```sh
./install_deps.sh
./build.sh
cmake -B build
cmake --build build
```

# Usage

The python library is pretty simple, with a `Client` class that has an `infer` method.

The `Client` class is used with the help of a context manager to automatically cleanup C++ objects:

```py
import kaldi_asr_client

with kaldi_asr_client.Client() as client:
   ...
```

For getting inferences on data, the `infer` method is used which takes a `list` of `bytes` as an argument and returns the corresponding inferences in order. Internally, the library runs the inference process in parallel on the GPU.

```py
import kaldi_asr_client

files = [
	"./data/1.wav",
	"./data/2.wav"
]

# Raw bytes corresponding to each file
wavs = []

for file in files:
	with open(file, "rb") as f:
		wavs.append(f.read())

with kaldi_asr_client.Client() as client:
    for index, inferred_text in enumerate(client.infer(wavs)):
        print(f"Inferred text for file {files[index]}: '{inferred_text}'")
```

Care must be taken to ensure that the file is opened in binary mode (`rb`).

Here is another example that takes WAV files as program arguments:

```py
from kaldi_asr_client import Client
import sys

wavs = []

for wav in sys.argv[1:]:
    with open(wav, "rb") as wav_fd:
        wavs.append((wav, wav_fd.read()))

# Using context manager to avoid memory leaks
with Client() as client:
    # Client.infer expects a list of WAV bytes objects
    wav_bytes = list(map(lambda item: item[1], wavs))

    for index, inference in enumerate(client.infer(wav_bytes)):
        wav_file = wavs[index][0]
        print(f"Inference for {wav_file}: {inference}")
```
