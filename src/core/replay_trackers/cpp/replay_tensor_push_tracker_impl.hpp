// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/tensor_push_tracker_base.hpp>

#include <cstdint>
#include <vector>

namespace core
{

// Replay has no peer to push to; mirrors ReplayMessageChannelTrackerImpl::send_message
// (logs and drops the payload rather than throwing).
class ReplayTensorPushTrackerImpl : public ITensorPushTrackerImpl
{
public:
    ReplayTensorPushTrackerImpl() = default;

    void update(int64_t monotonic_time_ns) override;
    void push(const std::vector<uint8_t>& payload) const override;
};

} // namespace core
