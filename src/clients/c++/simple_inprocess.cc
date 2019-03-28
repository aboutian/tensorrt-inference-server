// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <unistd.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include "src/core/request_inprocess.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  do {                                                             \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  } while (false)

namespace {

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-r [model repository absolute path]" << std::endl;
  std::cerr << "\t-m [model name]" << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string model_repository_path;
  std::string model_name;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vr:m:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'r':
        model_repository_path = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (model_repository_path.empty()) {
    Usage(argv, "-r must be used to specify model repository path");
  }
  if (model_name.empty()) {
    Usage(argv, "-m must be used to specify model name");
  }

  // Set the options for inference server and then create the
  // inference server object.
  std::unique_ptr<nic::InferenceServerContext::Options> server_options;
  FAIL_IF_ERR(
      nic::InferenceServerContext::Options::Create(&server_options),
      "unable to create inference server options");
  server_options->SetModelRepositoryPath(model_repository_path);

  std::unique_ptr<nic::InferenceServerContext> server_ctx;
  FAIL_IF_ERR(
      nic::InferenceServerContext::Create(&server_ctx, server_options),
      "unable to create inference server context");

  // Wait until the server is both live and ready.
  std::unique_ptr<nic::ServerHealthContext> health_ctx;
  FAIL_IF_ERR(
      nic::ServerHealthInProcessContext::Create(
          &health_ctx, server_ctx, verbose),
      "unable to create health context");

  while (true) {
    bool live, ready;
    FAIL_IF_ERR(health_ctx->GetLive(&live), "unable to get server liveness");
    FAIL_IF_ERR(health_ctx->GetReady(&ready), "unable to get server readiness");
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Check status of the server.
  std::unique_ptr<nic::ServerStatusContext> status_ctx;
  FAIL_IF_ERR(
      nic::ServerStatusInProcessContext::Create(
          &status_ctx, server_ctx, verbose),
      "unable to create status context");

  ni::ServerStatus server_status;
  FAIL_IF_ERR(
      status_ctx->GetServerStatus(&server_status),
      "unable to get server status");
  std::cout << "Server Status:" << std::endl;
  std::cout << server_status.DebugString() << std::endl;

  // Create an inference context for the model.
  std::unique_ptr<nic::InferContext> infer_ctx;
  FAIL_IF_ERR(
      nic::InferInProcessContext::Create(
          &infer_ctx, server_ctx, model_name, -1 /* model_version */, verbose),
      "unable to create inference context");

  return 0;
}
