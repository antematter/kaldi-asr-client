// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <grpc_client.h>

#include <cassert>
#include <functional>
#include <queue>
#include <signal.h>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef TRITON_KALDI_ASR_CLIENT_H_
#define TRITON_KALDI_ASR_CLIENT_H_

namespace tc = triton::client;
using TritonCallback = std::function<void(size_t, std::vector<std::string>)>;

class TritonASRClient {
  struct TritonClient {
    std::unique_ptr<tc::InferenceServerGrpcClient> triton_client;
  };

  std::string url_;
  std::string model_name_;

  std::vector<TritonClient> clients_;
  int nclients_;
  std::vector<uint8_t> chunk_buf_;
  int max_chunk_byte_size_;
  std::atomic<int> n_in_flight_;
  bool ctm_;
  int samps_per_chunk_;
  bool verbose_;

  TritonCallback infer_callback_;

  std::exception_ptr exception_ptr_;
  std::mutex exception_m_;

  bool streams_started_;

  void StreamCallback(tc::InferResult *result);

  void StartStreams(void);
  void StopStreams(void);

public:
  TritonASRClient(const std::string &url, const std::string &model_name,
                  const int ncontextes, bool ctm, bool verbose,
                  const TritonCallback &infer_callback_);

  void ResetClientContextes();
  void InferReset();
  void SendChunk(uint64_t corr_id, bool start_of_sequence, bool end_of_sequence,
                 float *chunk, int chunk_byte_size, uint64_t index);
  bool IsCallbacksDone();
  bool IsServerAlive();
};

void asr_signal_handler(int sig, siginfo_t *info, void *context);
#endif // TRITON_KALDI_ASR_CLIENT_H_
