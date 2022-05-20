// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/CNCLTools.h"
#include "paddle/fluid/distributed/collective/Types.h"

namespace paddle {
namespace distributed {

cnclReduceOp_t ToCNCLRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, cnclReduceOp_t> red_type = {
      {ReduceOp::MIN, cnclMin},
      {ReduceOp::MAX, cnclMax},
      {ReduceOp::SUM, cnclSum},
      {ReduceOp::PRODUCT, cnclProd},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(), true,
                    platform::errors::InvalidArgument(
                        "Invalid cncl reduction. Must be cnclMin | cnclMax | "
                        "cnclProd | cnclSum"));
  return it->second;
}

std::string SerializeCNCLCliqueId(const cnclCliqueId cnclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&cnclID);
  std::ostringstream oss;
  for (auto i = 0; i < CNCL_CLIQUE_ID_BYTES_SIZE; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

bool CheckTensorsInMluPlace(const std::vector<phi::DenseTensor>& tensors) {
  return std::all_of(tensors.cbegin(), tensors.cend(),
                     [&](const phi::DenseTensor& t) {
                       return platform::is_mlu_place(t.place());
                     });
}

}  //  namespace distributed
}  //  namespace paddle
