// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/live_trackers/schema_tracker.hpp"

#include <deviceio_trackers/exo_arms_tracker.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <schema/exo_arms_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using ExoArmsMcapChannels = McapTrackerChannels<ExoArmsOutputRecord, ExoArmsOutput>;
using ExoArmsSchemaTracker = SchemaTracker<ExoArmsOutputRecord, ExoArmsOutput>;

class LiveExoArmsTrackerImpl : public IExoArmsTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return SchemaTrackerBase::get_required_extensions();
    }
    static std::unique_ptr<ExoArmsMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                     std::string_view base_name);

    LiveExoArmsTrackerImpl(const OpenXRSessionHandles& handles,
                           const ExoArmsTracker* tracker,
                           std::unique_ptr<ExoArmsMcapChannels> mcap_channels);

    LiveExoArmsTrackerImpl(const LiveExoArmsTrackerImpl&) = delete;
    LiveExoArmsTrackerImpl& operator=(const LiveExoArmsTrackerImpl&) = delete;
    LiveExoArmsTrackerImpl(LiveExoArmsTrackerImpl&&) = delete;
    LiveExoArmsTrackerImpl& operator=(LiveExoArmsTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const ExoArmsOutputTrackedT& get_data() const override;

private:
    std::unique_ptr<ExoArmsMcapChannels> mcap_channels_;
    ExoArmsSchemaTracker m_schema_reader;
    ExoArmsOutputTrackedT m_tracked;
};

} // namespace core
