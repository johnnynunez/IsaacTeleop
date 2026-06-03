// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "replay_tensor_push_tracker_impl.hpp"

#include <iostream>

namespace core
{

void ReplayTensorPushTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
}

void ReplayTensorPushTrackerImpl::push(const std::vector<uint8_t>& /*payload*/) const
{
    std::cerr << "ReplayTensorPushTrackerImpl::push: ignored (no peer in replay mode)" << std::endl;
}

} // namespace core
