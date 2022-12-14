# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17..3.20)

add_library(kaldi-asr-parallel-client SHARED)
add_executable(grpc_connect_client grpc_connect_client.cc)

target_sources(kaldi-asr-parallel-client
  PRIVATE
    kaldi_asr_parallel_client.cc
    asr_client_imp.cc
)

set(inc_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/../../.build/client/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../.build/kaldi/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../.build/kaldi/tools/openfst/include
)

target_include_directories(kaldi-asr-parallel-client SYSTEM
  PRIVATE ${inc_dirs}
)
target_include_directories(kaldi-asr-parallel-client
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(grpc_connect_client PRIVATE ${inc_dirs})

target_compile_features(kaldi-asr-parallel-client PRIVATE cxx_std_17)
target_compile_options(kaldi-asr-parallel-client
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set(link_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/../../.build/kaldi/src/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../.build/client/lib
)

target_link_directories(kaldi-asr-parallel-client
  PRIVATE ${link_dirs}
)
target_link_directories(grpc_connect_client PRIVATE ${link_dirs})
target_link_libraries(kaldi-asr-parallel-client
  PRIVATE
    -lkaldi-base
    -lkaldi-util
    -lkaldi-matrix
    -lkaldi-feat
    -lkaldi-lat
    -lgrpcclient
    -lprotobuf
)
target_link_libraries(grpc_connect_client PRIVATE
    -lgrpcclient
    -lprotobuf
)
