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

#include "asr_client_imp.h"

#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-table.h"

#define RAISE_IF_ERR(X, MSG)                                                   \
  {                                                                            \
    tc::Error err = (X);                                                       \
    if (!err.IsOk()) {                                                         \
      std::stringstream ss;                                                    \
      ss << (MSG) << ": " << err;                                              \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  }

std::atomic<bool> asr_quit_(false);

void asr_signal_handler(int sig, siginfo_t *info, void *context) {
  if (sig == SIGINT) {
    asr_quit_ = true;
  }
}

void TritonASRClient::StreamCallback(tc::InferResult *result) {
  std::unique_ptr<tc::InferResult> result_ptr(result);
  RAISE_IF_ERR(result_ptr->RequestStatus(), "inference request failed");
  std::string request_id;
  RAISE_IF_ERR(result_ptr->Id(&request_id),
               "unable to get request id for response");
  uint64_t corr_id =
      std::stoi(std::string(request_id, 0, request_id.find("_")));

  bool end_of_stream = (request_id.back() == '1');
  if (!end_of_stream) {
    return;
  }

  std::vector<std::string> text;
  RAISE_IF_ERR(result_ptr->StringData(ctm_ ? "CTM" : "TEXT", &text),
               "unable to get TEXT or CTM output");
  infer_callback_(corr_id, text);

  n_in_flight_.fetch_sub(1, std::memory_order_relaxed);
};

void TritonASRClient::ResetClientContextes() {
  clients_.clear();

  for (int i = 0; i < nclients_; ++i) {
    clients_.emplace_back();
    TritonClient &client = clients_.back();

    RAISE_IF_ERR(tc::InferenceServerGrpcClient::Create(&client.triton_client,
                                                       url_, verbose_),
                 "unable to create triton client");
  }
}

void TritonASRClient::SendChunk(uint64_t corr_id, bool start_of_sequence,
                                bool end_of_sequence, float *chunk,
                                int chunk_byte_size, const uint64_t index) {
  assert(streams_started_);

  // Setting options
  tc::InferOptions options(model_name_);
  options.sequence_id_ = corr_id;
  options.sequence_start_ = start_of_sequence;
  options.sequence_end_ = end_of_sequence;
  options.request_id_ = std::to_string(corr_id) + "_" + std::to_string(index) +
                        "_" + (start_of_sequence ? "1" : "0") + "_" +
                        (end_of_sequence ? "1" : "0");

  // Initialize the inputs with the data.
  tc::InferInput *wave_data_ptr;
  std::vector<int64_t> wav_shape{1, samps_per_chunk_};
  RAISE_IF_ERR(
      tc::InferInput::Create(&wave_data_ptr, "WAV_DATA", wav_shape, "FP32"),
      "unable to create 'WAV_DATA'");
  std::shared_ptr<tc::InferInput> wave_data_in(wave_data_ptr);
  RAISE_IF_ERR(wave_data_in->Reset(), "unable to reset 'WAV_DATA'");
  uint8_t *wave_data = reinterpret_cast<uint8_t *>(chunk);
  if (chunk_byte_size < max_chunk_byte_size_) {
    std::memcpy(&chunk_buf_[0], chunk, chunk_byte_size);
    wave_data = &chunk_buf_[0];
  }
  RAISE_IF_ERR(wave_data_in->AppendRaw(wave_data, max_chunk_byte_size_),
               "unable to set data for 'WAV_DATA'");

  // Dim
  tc::InferInput *dim_ptr;
  std::vector<int64_t> shape{1, 1};
  RAISE_IF_ERR(tc::InferInput::Create(&dim_ptr, "WAV_DATA_DIM", shape, "INT32"),
               "unable to create 'WAV_DATA_DIM'");
  std::shared_ptr<tc::InferInput> dim_in(dim_ptr);
  RAISE_IF_ERR(dim_in->Reset(), "unable to reset WAVE_DATA_DIM");
  int nsamples = chunk_byte_size / sizeof(float);
  RAISE_IF_ERR(dim_in->AppendRaw(reinterpret_cast<uint8_t *>(&nsamples),
                                 sizeof(int32_t)),
               "unable to set data for WAVE_DATA_DIM");

  std::vector<tc::InferInput *> inputs = {wave_data_in.get(), dim_in.get()};

  std::vector<const tc::InferRequestedOutput *> outputs;
  std::shared_ptr<tc::InferRequestedOutput> raw_lattice, text;
  outputs.reserve(2);
  if (end_of_sequence) {
    tc::InferRequestedOutput *raw_lattice_ptr;
    RAISE_IF_ERR(
        tc::InferRequestedOutput::Create(&raw_lattice_ptr, "RAW_LATTICE"),
        "unable to get 'RAW_LATTICE'");
    raw_lattice.reset(raw_lattice_ptr);
    outputs.push_back(raw_lattice.get());

    // Request the TEXT results only when required for printing
    tc::InferRequestedOutput *text_ptr;
    RAISE_IF_ERR(
        tc::InferRequestedOutput::Create(&text_ptr, ctm_ ? "CTM" : "TEXT"),
        "unable to get 'TEXT' or 'CTM'");
    text.reset(text_ptr);
    outputs.push_back(text.get());
  }

  if (start_of_sequence) {
    n_in_flight_.fetch_add(1, std::memory_order_consume);
  }

  TritonClient *client = &clients_[corr_id % nclients_];
  // tc::InferenceServerGrpcClient& triton_client = *client->triton_client;
  RAISE_IF_ERR(
      client->triton_client->AsyncStreamInfer(options, inputs, outputs),
      "unable to run model");
}

void TritonASRClient::StartStreams() {
  assert(!streams_started_);

  for (auto &client : clients_) {
    RAISE_IF_ERR(client.triton_client->StartStream(
                     [&](tc::InferResult *result) {
                       try {
                         StreamCallback(result);
                       } catch (const std::exception &ex) {
                         std::lock_guard<std::mutex> lk(exception_m_);

                         if (!exception_ptr_) {
                           exception_ptr_ = std::current_exception();
                         }
                       }
                     },
                     false),
                 "unable to establish a streaming connection to server");
  }

  streams_started_ = true;
}

void TritonASRClient::StopStreams() {
  for (auto &client : clients_) {
    (*client.triton_client).StopStream();
  }

  streams_started_ = false;
}

TritonASRClient::TritonASRClient(const std::string &url,
                                 const std::string &model_name,
                                 const int nclients, bool ctm, bool verbose,
                                 const TritonCallback &infer_callback)
    : url_(url), model_name_(model_name), nclients_(nclients), ctm_(ctm),
      verbose_(verbose), infer_callback_(infer_callback) {
  nclients_ = std::max(nclients_, 1);

  ResetClientContextes();

  inference::ModelMetadataResponse model_metadata;
  RAISE_IF_ERR(
      clients_[0].triton_client->ModelMetadata(&model_metadata, model_name),
      "unable to get model metadata");

  for (const auto &in_tensor : model_metadata.inputs()) {
    if (in_tensor.name().compare("WAV_DATA") == 0) {
      samps_per_chunk_ = in_tensor.shape()[1];
    }
  }

  max_chunk_byte_size_ = samps_per_chunk_ * sizeof(float);
  chunk_buf_.resize(max_chunk_byte_size_);
  n_in_flight_.store(0);
}

int TritonASRClient::WaitForCallbacks() {
  while (n_in_flight_.load(std::memory_order_consume)) {
    std::lock_guard<std::mutex> lk(exception_m_);

    if (exception_ptr_) {
      std::rethrow_exception(exception_ptr_);
    }

    if (asr_quit_) {
      return 1;
    }

    usleep(100);
  }

  return 0;
}

void TritonASRClient::InferReset() {
  StopStreams();
  exception_ptr_ = nullptr;
  n_in_flight_.store(0);
  asr_quit_ = false;

  ResetClientContextes();
  StartStreams();
}
