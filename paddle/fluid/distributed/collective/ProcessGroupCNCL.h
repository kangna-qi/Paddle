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

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/device/mlu/mlu_info.h"

#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"
#include "paddle/fluid/platform/device/mlu/mlu_stream.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"

#if defined(PADDLE_WITH_CNCL)
#include <cncl.h>
#include "paddle/fluid/distributed/collective/CNCLTools.h"
#endif

constexpr const char* CNCL_BACKEND_NAME = "CNCL";

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;
using MLUStream = platform::stream::MLUStream;
using MLUDeviceContext = paddle::platform::MLUDeviceContext;

class ProcessGroupCNCL : public ProcessGroup {
 public:
  class CNCLTask : public ProcessGroup::Task,
                   public std::enable_shared_from_this<CNCLTask> {
   public:
    CNCLTask(const std::vector<Place>& places, int rank, CommType CommType,
             const std::vector<phi::DenseTensor>& inputs);

    bool IsCompleted();

    void SynchronizeStreams();

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void Synchronize();

    void SetOutputs(std::vector<phi::DenseTensor>& outputs);  // NOLINT

    virtual ~CNCLTask();

    std::vector<MLUEventManager> control_events_;
    std::vector<phi::DenseTensor> barrierTensors_;

   protected:
    std::vector<Place> places_;
    std::vector<std::shared_ptr<CNCLCommManager>> cnclComms_;
    std::shared_ptr<std::vector<phi::DenseTensor>> outputs_;

   private:
  };

  ProcessGroupCNCL(const std::shared_ptr<Store>& store, int rank, int size,
                   const platform::Place& place, int gid);

  const std::string GetBackendName() const override {
    return std::string(CNCL_BACKEND_NAME);
  }

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>& tensors, int dst_rank) override;

  std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>& tensors, int src_rank) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>& in,
      std::vector<phi::DenseTensor>& out) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ReduceOptions& opts) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ScatterOptions&) override;

 protected:
  virtual std::shared_ptr<ProcessGroupCNCL::CNCLTask> CreateTask(
      std::vector<Place> places, int rank, CommType opType,
      const std::vector<phi::DenseTensor>& inputs);

 protected:
  std::shared_ptr<Store> store_;
  std::shared_ptr<CNCLCommManager> cncl_comm_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<CNCLCommManager>>>
      places_to_cnclcomm_;

  std::unordered_map<std::string, std::vector<MLUEventManager>>
      places_to_events_;

  std::unordered_map<std::string,
                     std::vector<std::unique_ptr<MLUDeviceContext>>>
      places_to_ctx_;

  std::set<int> used_place_ids_;

 private:
  void BroadcastUniqueCNCLID(std::vector<cnclCliqueId>& cncl_ids);  // NOLINT

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn, CommType op_type);

  template <typename Fn>
  void Collective(const phi::DenseTensor*, phi::DenseTensor*, Fn fn,
                  CommType op_type);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> PointToPoint(
      std::vector<phi::DenseTensor>& tensors,  // NOLINT
      Fn fn, int dst_rank, CommType op_type);

  void CreateCNCLManagerCache(const std::string& places_key,
                              const std::vector<Place>& places);
};

}  //  namespace distributed
}  //  namespace paddle
