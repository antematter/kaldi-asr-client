#include "asr_client_imp.h"
#include "feat/wave-reader.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <random>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <vector>

constexpr auto URL = "localhost:8001";
constexpr auto MODEL = "kaldi_online";

enum {
  CHUNK_LENGTH = 8160,
  NCLIENTS = 10,
};

struct client {
  std::vector<kaldi::WaveData> inputs;
  std::vector<std::string> outputs;
  size_t expected_inputs;
  size_t iter_idx;
};

struct membuf : std::streambuf {
  membuf(char *begin, char *end) { this->setg(begin, begin, end); }
};

int feed_wav(TritonASRClient &asr_client, kaldi::WaveData &wave_data) {
  uint64_t index = 0;
  int32 offset = 0;

  while (true) {
    kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);

    int32 samp_remaining = data.Dim() - offset;
    int32 num_samp = std::min(static_cast<int32>(CHUNK_LENGTH), samp_remaining);

    bool is_last_chunk = (CHUNK_LENGTH >= samp_remaining);
    bool is_first_chunk = (offset == 0);

    kaldi::SubVector<kaldi::BaseFloat> wave_part(data, offset, num_samp);

    std::cerr << "Sending chunk " << index << " with " << num_samp << " samples"
              << '\n'
              << "is_first: " << is_first_chunk
              << ", is_last: " << is_last_chunk << '\n';

    asr_client.SendChunk(1, is_first_chunk, is_last_chunk, wave_part.Data(),
                         wave_part.SizeInBytes(), index++);

    offset += num_samp;

    if (is_last_chunk) {
      break;
    }
  }

  return 0;
}

extern "C" {
struct client *client_alloc(void) {
  struct client *client = new struct client;
  return client;
}

void client_destroy(struct client *client) { delete client; }

int client_infer_begin(struct client *client, size_t len) {
  client->iter_idx = 0;
  client->expected_inputs = len;

  client->inputs.clear();
  client->outputs.clear();

  client->inputs.reserve(client->expected_inputs);
  client->outputs.reserve(client->expected_inputs);

  return 0;
}

int client_infer_feed(struct client *client, char *bytes, size_t len) {
  membuf sbuf(bytes, bytes + len);
  std::istream is(&sbuf);

  client->inputs.emplace_back();
  client->inputs[client->inputs.size() - 1].Read(is);

  return 0;
}

int client_infer_perform(struct client *client) {
  if (client->expected_inputs != client->inputs.size()) {
    std::cerr << "Expected " << client->expected_inputs << " inputs but got "
              << client->inputs.size() << "!\n";
    std::exit(1);
  }

  if (client->inputs.size() == 0) {
    std::cerr << "No inputs fed!" << '\n';
    std::exit(1);
  }

  kaldi::WaveData &wave_data = client->inputs[0];

  float samp_freq = wave_data.SampFreq();
  double duration = wave_data.Duration();

  std::cout << "Loaded file with frequency " << samp_freq << "hz, duration "
            << duration << '\n';

  TritonASRClient asr_client(URL, MODEL, NCLIENTS, true, false, false,
                             samp_freq);

  feed_wav(asr_client, wave_data);
  asr_client.WaitForCallbacks();

  if (client->outputs.size() != client->inputs.size()) {
    std::cerr << "Outputs (" << client->outputs.size()
              << ") not equal to Inputs (" << client->inputs.size() << ")\n";
    std::exit(1);
  }

  return 0;
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
}

int main(int argc, char *const argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " FILE" << '\n';
    return 1;
  }

  std::ifstream istrm(argv[1], std::ios::binary);
  kaldi::WaveData wave_data;

  wave_data.Read(istrm);

  float samp_freq = wave_data.SampFreq();
  double duration = wave_data.Duration();

  std::cout << "Loaded file with frequency " << samp_freq << "hz, duration "
            << duration << '\n';

  TritonASRClient asr_client(URL, MODEL, NCLIENTS, true, false, false,
                             samp_freq);

  feed_wav(asr_client, wave_data);

  asr_client.WaitForCallbacks();
}
