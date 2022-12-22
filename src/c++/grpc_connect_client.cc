#include <grpc_client.h>

namespace tc = triton::client;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    return 1;
  }

  char *url = argv[1];

  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  tc::Error error =
      tc::InferenceServerGrpcClient::Create(&client, argv[1], false);

  if (!error.IsOk()) {
    std::cerr << "failed to create client: " << error << '\n';
    return 1;
  }

  bool is_live;
  error = (*client).IsServerLive(&is_live);

  if (!error.IsOk()) {
    std::cerr << "failed to check live status: " << error << '\n';
    return 1;
  }

  if (!is_live) {
    std::cerr << "server not live" << '\n';
    return 1;
  }

  return 0;
}
