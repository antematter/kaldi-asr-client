# kaldi-asr-client

# Building

The following commands will create a `pip install`-able wheel at `dist/kaldi_asr_client-*.whl`:

```sh
./scripts/install_deps.sh # Install system deps
./scripts/build.sh # Build kaldi and the triton library, will take a while
./scripts/setup_prebuilts.sh # Make all built libraries available for Python
python setup.py bdist_wheel
```

# Usage

The python library contains a `Client` class that has an `infer` method for getting inferences on data.

The `Client` class is used with the help of a context manager to automatically cleanup C++ objects. It takes the following keyword arguments:

* `samp_freq`: The sample frequency of the data. Must be consistent across all WAV files.
* `servers`: A `list` of server addresses to connect to. Multiple addresses can be passed for load balancing (eg. `["localhost:8001", "localhost:8002"]`). Default is `["localhost:8001"]`.
* `model_name`: Name of the model to be passed to Triton server. Default is `"kaldi_online"`.
* `ncontextes`: Number of clients to run in parallel per GPU (server). Default is `10`. **This spawns `ncontextes` number of threads per GPU, meaning that for 10 parallel clients and 2 GPUs, a total of 20 threads will be created.**
* `chunk_length`: Size of each chunk of the `WAV_DATA` sent to the server. Default is `8160`.
* `verbose`: Enable debug output. Default is `False`.

```py
import kaldi_asr_client

# Only samp_freq is explicitly set here. Rest of the arguments have a default value set.
with kaldi_asr_client.Client(samp_freq=16000) as client:
   ...
```

In cases of any errors, a simple `Exception` will be raised, detailing the error that occurred. For example:

```py
Traceback (most recent call last):
  File "/home/muhammad/kaldi-asr-client/ex.py", line 22, in <module>
    for index, inference in enumerate(client.infer(wav_bytes)):
  File "/home/muhammad/kaldi-asr-client/kaldi_asr_client.py", line 96, in infer
    self.client_infer_perform(self.client)
  File "/home/muhammad/kaldi-asr-client/kaldi_asr_client.py", line 80, in wrap_exc
    raise Exception(self.client_last_error(self.client).decode())
Exception: Non-uniform sample frequency! Expected 22050, got 16000
```

Another example, in case the server is not reachable:

```py
Traceback (most recent call last):
  File "/home/muhammad/kaldi-asr-client/ex.py", line 11, in <module>
    with Client(
  File "/home/muhammad/kaldi-asr-client/kaldi_asr_client.py", line 67, in __init__
    self.client_set_config(
  File "/home/muhammad/kaldi-asr-client/kaldi_asr_client.py", line 80, in wrap_exc
    raise Exception(self.client_last_error(self.client).decode())
Exception: unable to get model metadata: failed to connect to all addresses; last error: UNKNOWN: Failed to connect to remote host: Connection refused
```

For getting inferences on data, the `infer` method is used which takes a `list` of `bytes` as an argument and returns the corresponding inferences in order. Internally, the library runs the inference process in parallel on the available GPUs.

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

# Assuming sample rate of 16000
with kaldi_asr_client.Client(samp_freq=16000) as client:
    for index, inferred_text in enumerate(client.infer(wavs)):
        print(f"Inferred text for file {files[index]}: '{inferred_text}'")
```

Care must be taken to ensure that the file is opened in binary mode (`rb`).

Here is another example that takes WAV files as program arguments and explicitly sets all arguments:

```py
from kaldi_asr_client import Client
import sys

wavs = []

for wav in sys.argv[1:]:
    with open(wav, "rb") as wav_fd:
        wavs.append((wav, wav_fd.read()))

# Using context manager to avoid memory leaks
with Client(
    samp_freq=16000,
    servers=["localhost:8001"],
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
```
