#include "asr_client_imp.h"
#include "feat/wave-reader.h"
#include <assert.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <random>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <vector>

struct client {
  bool verbose;
  int chunk_length;
  float samp_freq;
  std::vector<std::unique_ptr<TritonASRClient>> clients;
  std::vector<kaldi::WaveData> inputs;
  std::vector<std::string> outputs;
  size_t expected_inputs;
  size_t iter_idx;
  std::string last_error;
};

struct membuf : std::streambuf {
  membuf(char *begin, char *end) { this->setg(begin, begin, end); }
};

template <typename Callable, typename... Args>
int invoke_wrap_exception(Callable fn, struct client *client, Args &...args) {
  try {
    return fn(client, args...);
  } catch (const std::exception &e) {
    client->last_error = e.what();
    return -1;
  } catch (...) {
    std::cerr << "Thrown exception doesn't inherit from std::exception!\n";
    std::abort();
  }
}

/* Returns 1 on EINTR */
int WaitForCallbacks(std::vector<std::unique_ptr<TritonASRClient>> &clients) {
  while (true) {
    bool all_done = true;

    for (auto &asr_client : clients) {
      if (!(*asr_client).IsCallbacksDone()) {
        all_done = false;

        if (!(*asr_client).IsServerAlive()) {
          throw std::runtime_error("Server is not live");
        }
      }
    }

    if (all_done) {
      break;
    }

    if (usleep(1000) == -1 && errno == EINTR) {
      return 1;
    }
  }

  return 0;
}

int feed_wav(TritonASRClient &asr_client, kaldi::WaveData &wave_data,
             const int chunk_length, const size_t corr_id, bool verbose) {
  uint64_t index = 0;
  int32 offset = 0;

  while (true) {
    kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);

    int32 samp_remaining = data.Dim() - offset;
    int32 num_samp = std::min(chunk_length, samp_remaining);

    bool is_last_chunk = (chunk_length >= samp_remaining);
    bool is_first_chunk = (offset == 0);

    kaldi::SubVector<kaldi::BaseFloat> wave_part(data, offset, num_samp);

    if (verbose) {
      std::cerr << "Sending chunk " << index << " with " << num_samp
                << " samples" << '\n'
                << "is_first: " << is_first_chunk
                << ", is_last: " << is_last_chunk << '\n';
    }

    asr_client.SendChunk(corr_id, is_first_chunk, is_last_chunk,
                         wave_part.Data(), wave_part.SizeInBytes(), index++);

    offset += num_samp;

    if (is_last_chunk) {
      break;
    }
  }

  return 0;
}

int client_infer_begin_(struct client *client, size_t len) {
  for (auto &asr_client : client->clients) {
    (*asr_client).InferReset();
  }

  client->iter_idx = 0;
  client->expected_inputs = len;

  client->inputs.clear();
  client->outputs.clear();

  client->inputs.reserve(client->expected_inputs);
  client->outputs.reserve(client->expected_inputs);

  for (size_t i = 0; i < client->expected_inputs; i++) {
    client->outputs.emplace_back();
  }

  return 0;
}

int client_infer_feed_(struct client *client, char *bytes, size_t len) {
  membuf sbuf(bytes, bytes + len);
  std::istream is(&sbuf);

  client->inputs.emplace_back();
  client->inputs[client->inputs.size() - 1].Read(is);

  return 0;
}

int client_infer_perform_(struct client *client) {
  assert(client->clients.size() > 0);

  if (client->expected_inputs != client->inputs.size()) {
    std::stringstream ss;
    ss << "Expected " << client->expected_inputs << " inputs but got "
       << client->inputs.size() << "inputs";

    throw std::runtime_error(ss.str());
  }

  if (client->inputs.size() == 0) {
    throw std::runtime_error("No inputs fed");
  }

  assert(client->outputs.size() == client->inputs.size());

  for (auto &wave_data : client->inputs) {
    if (wave_data.SampFreq() != client->samp_freq) {
      std::stringstream ss;
      ss << "Non-uniform sample frequency! Expected " << client->samp_freq
         << ", got " << wave_data.SampFreq();

      throw std::runtime_error(ss.str());
    }
  }

  for (size_t i = 0, corr_id = 1; i < client->inputs.size(); i++, corr_id++) {
    feed_wav(*client->clients[i % client->clients.size()], client->inputs[i],
             client->chunk_length, corr_id, client->verbose);
  }

  return WaitForCallbacks(client->clients);
}

int client_set_config_(struct client *client, float samp_freq, char *servers[],
                       char *model_name, int ncontextes, int chunk_length,
                       bool ctm, bool verbose) {
  auto infer_callback = [client](size_t corr_id,
                                 std::vector<std::string> text) {
    assert(corr_id > 0);

    client->outputs[corr_id - 1] = text[0];
  };

  if (!(*servers)) {
    throw std::runtime_error("No server addresses passed!");
  }

  client->verbose = verbose;
  client->chunk_length = chunk_length;
  client->samp_freq = samp_freq;

  while (*servers) {
    std::unique_ptr<TritonASRClient> asr_client(
        new TritonASRClient(*servers++, model_name, ncontextes, ctm,
                            client->verbose, TritonCallback(infer_callback)));
    client->clients.push_back(std::move(asr_client));
  }

  return 0;
}

extern "C" {
struct client *client_alloc(void) { return new struct client; }

int client_set_config(struct client *client, float samp_freq, char *servers[],
                      char *model_name, int ncontextes, int chunk_length,
                      bool ctm, bool verbose) {
  return invoke_wrap_exception(client_set_config_, client, samp_freq, servers,
                               model_name, ncontextes, chunk_length, ctm,
                               verbose);
}

int client_infer_begin(struct client *client, size_t len) {
  return invoke_wrap_exception(client_infer_begin_, client, len);
}

int client_infer_feed(struct client *client, char *bytes, size_t len) {
  return invoke_wrap_exception(client_infer_feed_, client, bytes, len);
}

int client_infer_perform(struct client *client) {
  return invoke_wrap_exception(client_infer_perform_, client);
}

/* Iterator since it's not possible to return non-trivial types without using
 * libpython. */
const char *client_infer_output(struct client *client) {
  if (client->iter_idx >= client->outputs.size()) {
    client->iter_idx = 0;
    return NULL;
  }

  return client->outputs[client->iter_idx++].c_str();
}

const char *client_last_error(struct client *client) {
  return client->last_error.c_str();
}

void client_destroy(struct client *client) { delete client; }
}
