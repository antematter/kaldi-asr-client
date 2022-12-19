#!/bin/bash 

# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

: ${GPU:=0}
: ${GRPC_PORT:=8001}

# Start Triton server 
docker run --rm \
   --gpus device="$GPU" \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -p"$GRPC_PORT":8001 \
   --entrypoint tritonserver \
   --name "trt_server_asr_$GPU" \
   -v $PWD/data:/data \
   -v $PWD/model-repo:/mnt/model-repo \
   --mount type=bind,source="$PWD/model-repo/kaldi_online/config.pbtxt",target="/workspace/model-repo/kaldi_online/config.pbtxt" \
   triton_kaldi_server \
   --model-repo=/workspace/model-repo # --log-verbose 1 --log-info 1
