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

#pragma once

#include <error.h>
#include <string>

#include "boost/variant.hpp"
#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"
#include "paddle/fluid/platform/device/mlu/mlu_info.h"

namespace paddle {
namespace distributed {

class MLUEventManager {
 public:
  MLUEventManager() = default;

  ~MLUEventManager() {
    if (is_created_) {
      platform::MLUDeviceGuard guard(device_index_);
      platform::MLUEventDestroy(event_);
    }
  }

  MLUEventManager(const MLUEventManager&) = delete;
  MLUEventManager& operator=(const MLUEventManager&) = delete;

  MLUEventManager(MLUEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }

  MLUEventManager& operator=(MLUEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
    return *this;
  }

  bool IsCreated() const { return is_created_; }
  bool DeviceId() const { return device_index_; }
  mluEventHandle GetRawMluEvent() const { return event_; }

  void Record(const paddle::platform::MLUDeviceContext& ctx) {
    auto device_index = ctx.GetPlace().device;
    if (!is_created_) {
      CreateEvent(device_index);
    }
    PADDLE_ENFORCE_EQ(device_index, device_index_,
                      platform::errors::PreconditionNotMet(
                          "MLUDeviceContext's device %d does not match"
                          "Event's device %d",
                          device_index, device_index_));

    platform::MLUDeviceGuard guard(device_index_);
    platform::MLUEventRecord(event_, ctx.stream());
  }

  bool Query() const {
    cnrtStatus err = platform::MLUEventQuery(event_);
    if (err == cnrtSuccess) {
      return true;
    } else if (err == cnrtErrorNotReady) {
      return false;
    } else {
      PADDLE_ENFORCE_MLU_SUCCESS(err);
      return false;
    }
  }

  void Synchronize() const {
    if (is_created_) {
      platform::MLUWaitEvent(event_);
    }
  }

  void Block(const paddle::platform::MLUDeviceContext& ctx) const {
    if (is_created_) {
      auto device_index = ctx.GetPlace().device;
      PADDLE_ENFORCE_EQ(device_index, device_index_,
                        platform::errors::PreconditionNotMet(
                            "MLUDeviceContext's device %d does not match"
                            "Event's device %d",
                            device_index, device_index_));
      platform::MLUDeviceGuard guard(device_index_);
      platform::MLUStreamWaitEvent(event_, ctx.stream(), 0);
    }
  }

 private:
  unsigned int flags_ = CNRT_NOTIFIER_DEFAULT;
  bool is_created_{false};
  mluEventHandle event_{};
  int8_t device_index_{0};

 private:
  void CreateEvent(int device_index) {
    device_index_ = device_index;
    platform::MLUDeviceGuard guard(device_index);
    platform::MLUEventCreateWithFlag(&event_, flags_);
    is_created_ = true;
  }
};

class CNCLCommManager {
 public:
  explicit CNCLCommManager(cnclComm_t cnclComm) : cncl_comm_(cnclComm) {}

  CNCLCommManager() : CNCLCommManager(nullptr) {}

  ~CNCLCommManager() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cncl_comm_) {
      cnclFreeComm(cncl_comm_);
    }
  }

  static std::shared_ptr<CNCLCommManager> Create(int num_ranks, int rank,
                                                 int dev_id,
                                                 cnclCliqueId_t comm_id) {
    auto cncl_manager = std::make_shared<CNCLCommManager>();
    cnclComm_t cncl_comm;
    int dev_list[] = {dev_id};
    int rank_list[] = {rank};
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnclInitComms(&cncl_comm, 1, dev_list, rank_list, num_ranks, comm_id));

    cncl_manager->cncl_id_ = comm_id;
    cncl_manager->rank_ = rank;
    cncl_manager->dev_id_ = dev_id;
    cncl_manager->cncl_comm_ = cncl_comm;
    return cncl_manager;
  }

  cnclCliqueId_t GetCnclId() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return cncl_id_;
  }

  cnclComm_t GetCnclComm() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return cncl_comm_;
  }

  CNCLCommManager(const CNCLCommManager&) = delete;
  CNCLCommManager& operator=(const CNCLCommManager&) = delete;
  CNCLCommManager& operator=(CNCLCommManager&& other) = delete;

  CNCLCommManager(CNCLCommManager&& other) {
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(cncl_comm_, other.cncl_comm_);
  }

 protected:
  cnclComm_t cncl_comm_;
  cnclCliqueId_t cncl_id_;
  int rank_;
  int dev_id_;
  mutable std::mutex mutex_;
};

cnclReduceOp_t ToCNCLRedType(ReduceOp reduction);
std::string SerializeCNCLCliqueId(const cnclCliqueId cnclID);
bool CheckTensorsInMluPlace(const std::vector<phi::DenseTensor>& tensors);

}  // namespace distributed
}  // namespace paddle
