# kaldi-asr-client

# Usage

The python library is pretty simple, with a `Client` class that has an `infer` method.

The `Client` class is used with the help of a context manager to automatically cleanup C++ objects:

```
import kaldi_asr_client

with kaldi_asr_client.Client() as client:
   ...
```

For getting inferences on data, the `infer` method is used which takes a `list` of `bytes` as an argument and returns the corresponding inferences in order:

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

Care must be taken to ensure that the file is opened in binary mode (`rb`)
