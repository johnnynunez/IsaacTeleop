// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tracker.hpp"

namespace core
{

struct ExoArmsOutputTrackedT;

// Abstract base interface for ExoArmsTracker implementations.
class IExoArmsTrackerImpl : public ITrackerImpl
{
public:
    virtual const ExoArmsOutputTrackedT& get_data() const = 0;
};

} // namespace core
