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

#include "paddle/fluid/distributed/collective/ProcessGroupCNCL.h"
#include "paddle/fluid/distributed/collective/Common.h"
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

DECLARE_bool(cncl_blocking_wait);
// DECLARE_bool(use_stream_safe_mlu_allocator);

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

void SyncDefaultStream(
    const std::vector<Place>& places,
    std::vector<MLUEventManager>& cnclEvents,                   // NOLINT
    std::vector<std::unique_ptr<MLUDeviceContext>>& dev_ctx) {  // NOLINT
  for (size_t i = 0; i < places.size(); ++i) {
    auto* default_ctx = static_cast<platform::MLUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places[i]));
    cnclEvents[i].Record(*default_ctx);
    cnclEvents[i].Block(*dev_ctx[i]);
  }
}

std::shared_ptr<ProcessGroupCNCL::CNCLTask> ProcessGroupCNCL::CreateTask(
    std::vector<Place> places, int rank, CommType comm_type,
    const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupCNCL::CNCLTask>(places, rank, comm_type,
                                                      inputs);
}

ProcessGroupCNCL::CNCLTask::CNCLTask(
    const std::vector<Place>& places, int rank, CommType CommType,
    const std::vector<phi::DenseTensor>& inputs)
    : Task(rank, inputs, CommType), places_(places) {
  control_events_.resize(places.size());
  cnclComms_.resize(places.size());
}

ProcessGroupCNCL::CNCLTask::~CNCLTask() {}

void ProcessGroupCNCL::CNCLTask::SetOutputs(
    std::vector<phi::DenseTensor>& outputs) {  // NOLINT
  outputs_ = std::make_shared<std::vector<phi::DenseTensor>>(outputs);
}

void ProcessGroupCNCL::CNCLTask::SynchronizeStreams() {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto* default_ctx = static_cast<platform::MLUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(places_[i]));
    platform::MLUStreamWaitEvent(control_events_[i].GetRawMluEvent(),
                                 default_ctx->stream(), 0);
  }
}

bool ProcessGroupCNCL::CNCLTask::IsCompleted() {
  for (size_t i = 0; i < places_.size(); ++i) {
    if (!control_events_[i].Query()) {
      return false;
    }
  }

  return true;
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupCNCL::CNCLTask::Wait(std::chrono::milliseconds timeout) {
  SynchronizeStreams();
  if (FLAGS_cncl_blocking_wait) {
    // NOTE(shenliang03): It will block host for sync
    while (!IsCompleted()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitBlockTImeout));
    }
  }

  if (!barrierTensors_.empty()) {
    // If we use the work to do barrier, we should block cpu
    for (auto& place : places_) {
      platform::MLUDeviceGuard mluGuard(place.GetDeviceId());
      platform::MLUDeviceSync();
    }
  }
  return true;
}

// Same as Wait
void ProcessGroupCNCL::CNCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupCNCL::ProcessGroupCNCL(const std::shared_ptr<Store>& store,
                                   int rank, int size,
                                   const platform::Place& place, int gid)
    : ProcessGroup(rank, size, place, gid), store_(store) {
  platform::SetMLUDeviceId(place_.device);
}

void ProcessGroupCNCL::BroadcastUniqueCNCLID(
    std::vector<cnclCliqueId>& cncl_ids) {  // NOLINT
  if (rank_ == 0) {
    for (size_t i = 0; i < cncl_ids.size(); i++) {
      auto key = "ProcessGroupCNCL/cncl_ids/" + std::to_string(gid_) + "/" +
                 std::to_string(i);
      auto cncl_id = std::vector<uint8_t>(
          reinterpret_cast<uint8_t*>(&cncl_ids[i]),
          reinterpret_cast<uint8_t*>(&cncl_ids[i]) + CNCL_CLIQUE_ID_BYTES_SIZE);
      store_->set(key, cncl_id);
    }
  } else {
    for (size_t i = 0; i < cncl_ids.size(); i++) {
      auto key = "ProcessGroupCNCL/cncl_ids/" + std::to_string(gid_) + "/" +
                 std::to_string(i);
      auto ret = store_->get(key);
      std::memcpy(&cncl_ids[i], ret.data(), ret.size());
    }
  }
}

// create CNCLManager cache for places_key
void ProcessGroupCNCL::CreateCNCLManagerCache(
    const std::string& places_key, const std::vector<Place>& places) {
  PADDLE_ENFORCE_EQ(places_key.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "Not able to create/get the CNCL Communicator since "
                        "the MLU place are not known"));

  std::vector<std::shared_ptr<CNCLCommManager>> cncl_comms;
  cncl_comms.resize(places.size());

  // using vector just for broadcast
  std::vector<cnclCliqueId> cncl_ids;
  cncl_ids.resize(1);
  auto& cncl_id = cncl_ids.front();

  for (auto& place : places) {
    used_place_ids_.insert(place.GetDeviceId());
  }

  if (rank_ == 0) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnclGetCliqueId(&cncl_id));
  }
  BroadcastUniqueCNCLID(cncl_ids);

  VLOG(3) << "init cncl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << places_key
          << ", cncl uniqueid: " << SerializeCNCLCliqueId(cncl_id);

  std::vector<std::unique_ptr<MLUDeviceContext>> dev_ctx;
  dev_ctx.resize(places.size());

  for (size_t i = 0; i < places.size(); ++i) {
    auto dev_id = places[i].GetDeviceId();
    platform::MLUDeviceGuard guard(dev_id);
    cncl_comms[i] =
        CNCLCommManager::Create(GetSize(), GetRank(), dev_id, &cncl_id);
    dev_ctx[i].reset(new MLUDeviceContext(places[i]));
  }

  std::vector<MLUEventManager> events;
  events.resize(places.size());

  // These caches will be useful to process sync/wait/communicate
  places_to_events_.emplace(places_key, std::move(events));
  places_to_cnclcomm_.emplace(places_key, std::move(cncl_comms));
  places_to_ctx_.emplace(places_key, std::move(dev_ctx));
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::Collective(
    std::vector<phi::DenseTensor>& inputs,
    std::vector<phi::DenseTensor>& outputs, Fn fn, CommType op_type) {
  const auto places = GetPlaceList(inputs);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_cnclcomm_.find(key) == places_to_cnclcomm_.end()) {
      CreateCNCLManagerCache(key, places);
    }
  }

  auto& cncl_comms = places_to_cnclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  auto task = CreateTask(places, rank_, op_type, inputs);
  task->SetOutputs(outputs);

  // construct uninitialize guard for device

  // if (FLAGS_use_stream_safe_mlu_allocator) {
  //  for (size_t i = 0; i < inputs.size(); ++i) {
  //    platform::MLUDeviceGuard guard(places[i].GetDeviceId());
  //    memory::RecordStream(inputs[i].Holder(),
  //                         places_to_ctx_[key][i]->stream());
  //  }
  //}

  for (size_t i = 0; i < inputs.size(); ++i) {
    platform::MLUDeviceGuard guard(places[i].GetDeviceId());
    const auto& cncl_stream = places_to_ctx_[key][i]->stream();
    fn(inputs[i], outputs[i], cncl_comms[i]->GetCnclComm(), cncl_stream);
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    platform::MLUDeviceGuard guard(places[i].GetDeviceId());
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

template <typename Fn>
void ProcessGroupCNCL::Collective(const phi::DenseTensor* in,
                                  phi::DenseTensor* out, Fn fn,
                                  CommType op_type) {
  std::vector<Place> places;
  places.push_back(in->place());
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_cnclcomm_.find(key) == places_to_cnclcomm_.end()) {
      CreateCNCLManagerCache(key, places);
    }
  }

  auto& cncl_comms = places_to_cnclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  // construct uninitialize guard for device

  // if (FLAGS_use_stream_safe_mlu_allocator) {
  //  platform::MLUDeviceGuard guard(places[i].GetDeviceId());
  //  memory::RecordStream(in->Holder(), places_to_ctx_[key][0]->stream());
  //}

  platform::MLUDeviceGuard guard(places[0].GetDeviceId());
  const auto& cncl_stream = places_to_ctx_[key][0]->stream();
  fn(in, out, cncl_comms[0]->GetCnclComm(), cncl_stream);
}

template <typename Fn>
std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::PointToPoint(
    std::vector<phi::DenseTensor>& tensors, Fn fn, int dst_rank,
    CommType op_type) {
  const auto places = GetPlaceList(tensors);
  const auto key = GetKeyFromPlaces(places);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (places_to_cnclcomm_.find(key) == places_to_cnclcomm_.end()) {
      CreateCNCLManagerCache(key, places);
    }
  }

  auto& cncl_comms = places_to_cnclcomm_[key];

  SyncDefaultStream(places, places_to_events_[key], places_to_ctx_[key]);

  auto task = CreateTask(places, rank_, op_type, tensors);

  // construct uninitialize guard for device

  // if (FLAGS_use_stream_safe_mlu_allocator) {
  //  for (size_t i = 0; i < tensors.size(); ++i) {
  //    platform::MLUDeviceGuard guard(places[i].GetDeviceId());
  //    memory::RecordStream(tensors[i].Holder(),
  //                         places_to_ctx_[key][i]->stream());
  //  }
  //}

  for (size_t i = 0; i < tensors.size(); ++i) {
    platform::MLUDeviceGuard guard(places[i].GetDeviceId());
    const auto& cncl_stream = places_to_ctx_[key][i]->stream();
    fn(tensors[i], cncl_comms[i]->GetCnclComm(), cncl_stream, dst_rank);
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    platform::MLUDeviceGuard guard(places[i].GetDeviceId());
    task->control_events_[i].Record(*places_to_ctx_[key][i]);
  }
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::AllReduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors, const AllreduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));
  return Collective(in_tensors, out_tensors,
                    [&](const phi::DenseTensor& input, phi::DenseTensor& output,
                        cnclComm_t comm, const mluStream& stream) {
                      return cnclAllReduce(
                          input.data(), output.data(), input.numel(),
                          platform::ToCNCLDataType(input.dtype()),
                          ToCNCLRedType(opts.reduce_op), comm, stream);
                    },
                    CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors, const BroadcastOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));

  return Collective(
      in_tensors, out_tensors,
      [&](phi::DenseTensor& input, phi::DenseTensor& output, cnclComm_t comm,
          const mluStream& stream) {
        const auto root =
            opts.source_rank * in_tensors.size() + opts.source_root;
        return cnclBroadcast(input.data(), output.data(), input.numel(),
                             platform::ToCNCLDataType(input.dtype()), root,
                             comm, stream);
      },
      CommType::BROADCAST);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::Barrier(
    const BarrierOptions& opts) {
  std::vector<phi::MLUPlace> places;

  if (!opts.place_ids.empty()) {
    for (auto place_id : opts.place_ids) {
      places.emplace_back(place_id);
    }
  } else if (!used_place_ids_.empty()) {
    for (auto place_id : used_place_ids_) {
      places.emplace_back(place_id);
    }
  } else {
    auto numMLUs = GetSize();
    int place_id = static_cast<int>(rank_ % numMLUs);
    places.emplace_back(place_id);
  }

  std::vector<phi::DenseTensor> barrierTensors;
  barrierTensors.reserve(places.size());

  for (auto& place : places) {
    platform::MLUDeviceGuard mluGuard(place.GetDeviceId());
    auto dt = full({1}, 0, phi::DataType::FLOAT32, phi::MLUPlace());
    barrierTensors.push_back(
        *std::dynamic_pointer_cast<phi::DenseTensor>(dt.impl()));
  }
  auto task = ProcessGroupCNCL::AllReduce(barrierTensors, barrierTensors);
  auto cncl_task = dynamic_cast<ProcessGroupCNCL::CNCLTask*>(task.get());
  cncl_task->barrierTensors_ = std::move(barrierTensors);
  return task;
}

void CheckTensorsInDifferentDevices(
    const std::vector<phi::DenseTensor>& tensors, const size_t num_devices) {
  PADDLE_ENFORCE_EQ(
      tensors.size() == 0, false,
      platform::errors::InvalidArgument("Tensor list must be nonempty."));
  PADDLE_ENFORCE_LE(
      tensors.size(), num_devices,
      platform::errors::InvalidArgument(
          "Tensor list mustn't be larger than the number of available MLUs."));

  std::set<Place> used_devices;

  for (const auto& t : tensors) {
    PADDLE_ENFORCE_EQ(platform::is_mlu_place(t.place()), true,
                      platform::errors::InvalidArgument(
                          "Tensors must be MLU and dense tensor."));

    const auto inserted = used_devices.insert(t.place()).second;
    PADDLE_ENFORCE_EQ(inserted, true,
                      platform::errors::InvalidArgument(
                          "Tensors must be on distinct MLU devices."));
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::Send(
    std::vector<phi::DenseTensor>& tensors, int dst_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task =
      PointToPoint(tensors,
                   [&](phi::DenseTensor& input, cnclComm_t comm,
                       const mluStream& stream, int dst_rank) {
                     return cnclSend(input.data(), input.numel(),
                                     platform::ToCNCLDataType(input.dtype()),
                                     dst_rank, comm, stream);
                   },
                   dst_rank, CommType::SEND);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::Recv(
    std::vector<phi::DenseTensor>& tensors, int src_rank) {
  CheckTensorsInDifferentDevices(tensors, static_cast<size_t>(GetSize()));

  auto task =
      PointToPoint(tensors,
                   [&](phi::DenseTensor& output, cnclComm_t comm,
                       const mluStream& stream, int src_rank) {
                     return cnclRecv(output.data(), output.numel(),
                                     platform::ToCNCLDataType(output.dtype()),
                                     src_rank, comm, stream);
                   },
                   src_rank, CommType::RECV);
  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::AllGather(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(out_tensors), true,
      platform::errors::InvalidArgument("All outputs should be in MluPlace."));
  return Collective(in_tensors, out_tensors,
                    [&](const phi::DenseTensor& input, phi::DenseTensor& output,
                        cnclComm_t comm, const mluStream& stream) {
                      return cnclAllGather(
                          input.data(), output.data(), input.numel(),
                          platform::ToCNCLDataType(input.dtype()), comm,
                          stream);
                    },
                    CommType::ALLGATHER);
}

void* GetPointerByOffset(void* raw_pointer, size_t offset,
                         experimental::DataType type) {
  if (type == experimental::DataType::FLOAT32) {
    return reinterpret_cast<void*>(reinterpret_cast<float*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::INT32) {
    return reinterpret_cast<void*>(reinterpret_cast<int32_t*>(raw_pointer) +
                                   offset);
  } else if (type == experimental::DataType::FLOAT16) {
    return reinterpret_cast<void*>(reinterpret_cast<int16_t*>(raw_pointer) +
                                   offset);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in cncl is not supported."));
  }
  return nullptr;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::AllToAll(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(out_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));
  return Collective(
      in_tensors, out_tensors,
      [&](phi::DenseTensor& input, phi::DenseTensor& output, cnclComm_t comm,
          const mluStream& stream) {
        size_t offset = 0;
        for (auto i = 0; i < size_; i++) {
          PADDLE_ENFORCE_MLU_SUCCESS(cnclSend(
              GetPointerByOffset(input.data(), offset, input.dtype()),
              input.numel() / size_, platform::ToCNCLDataType(input.dtype()), i,
              comm, stream));
          PADDLE_ENFORCE_MLU_SUCCESS(cnclRecv(
              GetPointerByOffset(output.data(), offset, input.dtype()),
              input.numel() / size_, platform::ToCNCLDataType(input.dtype()), i,
              comm, stream));
          offset += input.numel() / size_;
        }
      },
      CommType::ALLREDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::Reduce(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors, const ReduceOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));
  return Collective(
      in_tensors, out_tensors,
      [&](const phi::DenseTensor& input, phi::DenseTensor& output,
          cnclComm_t comm, const mluStream& stream) {
        PADDLE_ENFORCE_MLU_SUCCESS(cnclReduce(
            input.data(), output.data(), input.numel(),
            platform::ToCNCLDataType(input.dtype()),
            ToCNCLRedType(opts.reduce_op), opts.root_rank, comm, stream));
      },
      CommType::REDUCE);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupCNCL::Scatter(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors, const ScatterOptions& opts) {
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(in_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInMluPlace(out_tensors), true,
      platform::errors::InvalidArgument("All inputs should be in MluPlace."));
  return Collective(
      in_tensors, out_tensors,
      [&](phi::DenseTensor& input, phi::DenseTensor& output, cnclComm_t comm,
          const mluStream& stream) {
        size_t offset = 0;
        if (rank_ == opts.root_rank) {
          for (auto i = 0; i < size_; i++) {
            PADDLE_ENFORCE_MLU_SUCCESS(cnclSend(
                GetPointerByOffset(input.data(), offset, input.dtype()),
                input.numel() / size_, platform::ToCNCLDataType(input.dtype()),
                i, comm, stream));
            offset += input.numel() / size_;
          }
          PADDLE_ENFORCE_MLU_SUCCESS(
              cnclRecv(output.data(), input.numel() / size_,
                       platform::ToCNCLDataType(input.dtype()), opts.root_rank,
                       comm, stream));
        } else {
          PADDLE_ENFORCE_MLU_SUCCESS(
              cnclRecv(output.data(), input.numel() / size_,
                       platform::ToCNCLDataType(input.dtype()), opts.root_rank,
                       comm, stream));
        }
      },
      CommType::SCATTER);
}

}  //  namespace distributed
}  //  namespace paddle
