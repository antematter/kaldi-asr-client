# Kaldi ASR Client

# Table of contents

1. [Prerequisites](#prerequisites)
2. [Building](#building)
3. [Python Library Usage](#usage)
4. [Server Setup](#server-setup)
5. [Loading Custom Models](#custom-models)
6. [Known Issues](#known-issues)

# Prerequisites

- Ubuntu 22.04
- Docker
- Nvidia Driver Version: 520.61.05
- CUDA Version: 11.8

# Building

The following commands will create a `pip install`-able wheel at `dist/kaldi_asr_client-*.whl`:

```sh
./scripts/install_deps.sh # Install system deps
./scripts/build.sh # Build kaldi and the triton library, will take a while
./scripts/setup_prebuilts.sh # Make all built libraries available for Python
python setup.py bdist_wheel
```

# Usage

First of all, ensure that the `server_launch_daemon.sh` script is running. It is responsible for launching and restarting servers on-demand because of a memory leak in Kaldi:

```sh
./scripts/server_launch_daemon.sh
```

The python library contains a `Client` class that has an `infer` method for getting inferences on data.

The `Client` class is used with the help of a context manager to automatically cleanup C++ objects. It takes the following keyword arguments:

* `samp_freq`: The sample frequency of the data. Must be consistent across all WAV files.
* `servers`: A `list` of server addresses to connect to. Multiple addresses can be passed for load balancing (eg. `["localhost:8001", "localhost:8002"]`). Default is `["localhost:8001"]`.
* `model_name`: Name of the model to be passed to Triton server. Default is `"kaldi_online"`.
* `ncontextes`: Number of clients to run in parallel per GPU (server). Default is `10`. **This spawns `ncontextes` number of threads per GPU, meaning that for 10 parallel clients and 2 GPUs, a total of 20 threads will be created.**
* `chunk_length`: Size of each chunk of the `WAV_DATA` sent to the server. Default is `8160`.
* `ctm`: Get `CTM` output instead of `TEXT` from the server.
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

In-case the servers need to be (re)started with the `./scripts/server_launch_daemon.sh` daemon, the `restart_servers` method should be used before creating the `Client` object:

```py
from kaldi_asr_client import Client, restart_servers

SERVERS = ["localhost:8001", "localhost:8002"]

restart_servers(SERVERS)

with Client(...) as client:
    ...
```

# Server Setup

To run the triton server with the [LibriSpeech Model](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition#quick-start-guide), follow these steps:

1. Clone the repo

```sh
git clone https://github.com/NVIDIA/DeepLearningExamples.git --depth=1
cd DeepLearningExamples/Kaldi/SpeechRecognition
```

2. Build the server container

```sh
./scripts/docker/build.sh
```

3. Download and set up the pre-trained model and eval dataset into the `data/` folder

```sh
./scripts/docker/launch_download.sh
```

4. For launching the server, the launch script needs to be replaced with the one in this repo as the original script doesn't function well with multiple GPUs

```sh
cp ~/kaldi-asr-client/scripts/launch_server.sh ~/DeepLearningExamples/Kaldi/SpeechRecognition/scripts/docker/launch_server.sh
```

5. The script uses two environment variables, `$GPU` and `$GRPC_PORT` for the GPU to run the server on and the port to expose the server on respectively

By default, the script launches the server on GPU `0` and port `8001`, which is equivalent to running the following

```sh
GPU=0 GRPC_PORT=8001 ./scripts/docker/launch_server.sh
```

In-case the server needs to be run on another GPU, the environment variables can be tweaked accordingly. For example, the following snippet will run the server on GPU `1` and expose the server on port `8002`

```sh
GPU=1 GRPC_PORT=8002 ./scripts/docker/launch_server.sh
```

6. For getting CTM inferences, the `lattice_postprocessor_rxfilename` config parameter needs to be properly set up in `config.pbtxt`. For this, we need the `word_boundary.int` file aswell.

First of all, place the word boundary file into the model's directory along with the postprocessor config:

```sh
cp word_boundary.int ~/DeepLearningExamples/Kaldi/SpeechRecognition/data/models/LibriSpeech/
echo "--word-boundary-rxfilename=/data/models/LibriSpeech/word_boundary.int" > ~/DeepLearningExamples/Kaldi/SpeechRecognition/data/models/LibriSpeech/lattice_postprocess.conf
```

Finally, modify `Kaldi/SpeechRecognition/model-repo/kaldi_online/config.pbtxt` in accordance with the following diff:

```diff
--- a/Kaldi/SpeechRecognition/model-repo/kaldi_online/config.pbtxt
+++ b/Kaldi/SpeechRecognition/model-repo/kaldi_online/config.pbtxt
@@ -32,7 +32,7 @@ parameters [
   {
     key: "lattice_postprocessor_rxfilename"
     value { 
-      string_value: ""
+      string_value: "/data/models/LibriSpeech/lattice_postprocess.conf"
     }
   },
   {
```

Hence, the script can be used to run multiple servers at once, that can be passed into the `servers` argument of the client library.

# Custom Models

Essentially you just have to make two changes in the `DeepLearningExamples` repo:

* Replace `Kaldi/SpeechRecognition/model-repo/kaldi_online/config.pbtxt` with your own model's config.

* Place your actual model's directory in `Kaldi/SpeechRecognition/data/models/`.

The only thing to keep in mind is that any hardcoded paths should be changed to use `/data/models/$YOURMODELHERE` as their base directory in `config.pbtxt`. For example, the required changes might look like the following `diff`:

```diff
diff --git a/config.pbtxt b/config.pbtxt
index 1788c79..5ac38bb 100644
--- a/config.pbtxt
+++ b/config.pbtxt
@@ -8,31 +8,31 @@ model_transaction_policy {
 parameters: {
     key: "config_filename"
     value {
-	string_value:"/opt/models/my_model/conf/online.conf"
+	string_value:"/data/models/my_model/conf/online.conf"
     }
 }
 parameters: {
     key: "ivector_filename"
     value: {
-	string_value:"/opt/models/my_model/ivector/final.mat"
+	string_value:"/data/models/my_model/ivector/final.mat"
     }
 }
...
```

Now, we will change the default LibriSpeech model to another sample model, elucidating the above steps. Note that some steps here are specific to the sample model, such as moving the configs into the model's folder (They're usually present in there already, we just did that here as they were in a seperate tarball) and probably won't usually be required.

Model https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip

Config https://github.com/triton-inference-server/server/files/7734169/triton_vosk.tar.gz

* Download and extract both the files

```sh
tar xf triton_vosk.tar.gz
unzip vosk-model-ru-0.22.zip
```

* The configs extract into the `kaldi_online/2` directory, move them into the model's folder

```sh
mv kaldi_online/2/conf/* vosk-model-ru-0.22/conf
```

* Change hardcoded paths to point to paths mounted in the docker image

```sh
sed -i 's|/opt/sancom/models/kaldi_online/2|/data/models/vosk-model-ru-0.22|g' kaldi_online/config.pbtxt vosk-model-ru-0.22/conf/*
```

* Move the files into the Deeplearning examples repo

```sh
mv kaldi_online/config.pbtxt ~/DeepLearningExamples/Kaldi/SpeechRecognition/model-repo/kaldi_online/config.pbtxt
mv vosk-model-ru-0.22 ~/DeepLearningExamples/Kaldi/SpeechRecognition/data/models/
```

* Run the server

```sh
cd ~/DeepLearningExamples/Kaldi/SpeechRecognition
GPU=0 GRPC_PORT=8001 ./scripts/docker/launch_server.sh
```

Since we changed the `config.pbtxt` file inside the `kaldi_online` directory itself, there is no need to modify the `model_name` on the client side. It is likely possible to use a seperate directory for the model's `config.pbtxt` and pass that as the `model_name` on the client-side, but this has not been tested.

# Known Issues

* The `StopStream()` method of the Triton client library seems to have a delay proportional to the size of input when interrupting the inference operation mid-way. This causes a delay in cancelling the operations of the other active servers in-case multiple servers if one server crashes.

* There is a memory leak in [Kaldi](https://github.com/kaldi-asr/kaldi/issues/4814) that is currently worked around with the `./scripts/server_launch_daemon.sh` daemon that restarts servers on each run.

* The Triton client library needs to be rebuilt to link against Openssl 3 and get rid of the requirement for prebuilts. This is not possible until their CMake stuff is figured out.
