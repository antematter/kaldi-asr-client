Model https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip

Config https://github.com/triton-inference-server/server/files/7734169/triton_vosk.tar.gz

Download both the files

Extract both the files:

```sh
tar xf triton_vosk.tar.gz
unzip vosk-model-ru-0.22.zip
```

The configs extract into the `kaldi_online/2` directory, move them into the model's folder

```sh
mv kaldi_online/2/conf/* vosk-model-ru-0.22/conf
```

Change hardcoded paths to point to paths mounted in the docker image:

```sh
sed -i 's|/opt/sancom/models/kaldi_online/2|/data/models/vosk-model-ru-0.22|g' kaldi_online/config.pbtxt vosk-model-ru-0.22/conf/*
```

Move the files into the Deeplearning examples repo:

```sh
mv kaldi_online/config.pbtxt ~/DeepLearningExamples/Kaldi/SpeechRecognition/model-repo/kaldi_online/config.pbtxt
mv vosk-model-ru-0.22 ~/DeepLearningExamples/Kaldi/SpeechRecognition/data/models/
```

Run the server:

```sh
cd ~/DeepLearningExamples/Kaldi/SpeechRecognition
NVIDIA_VISIBLE_DEVICES=1 ./scripts/docker/launch_server.sh
```

Now the client can be used with any input as with the regular example model.
