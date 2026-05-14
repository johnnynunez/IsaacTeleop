// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_exo_arms_tracker_impl.hpp"

#include <mcap/recording_traits.hpp>
#include <schema/exo_arms_bfbs_generated.h>

namespace core
{

namespace
{

SchemaTrackerConfig make_exo_arms_tensor_config(const ExoArmsTracker* tracker)
{
    SchemaTrackerConfig cfg;
    cfg.collection_id = tracker->collection_id();
    cfg.max_flatbuffer_size = tracker->max_flatbuffer_size();
    cfg.tensor_identifier = "exo_arms";
    cfg.localized_name = "ExoArmsTracker";
    return cfg;
}

} // namespace

std::unique_ptr<ExoArmsMcapChannels> LiveExoArmsTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                  std::string_view base_name)
{
    return std::make_unique<ExoArmsMcapChannels>(writer, base_name, ExoArmsRecordingTraits::schema_name,
                                                 std::vector<std::string>(ExoArmsRecordingTraits::recording_channels.begin(),
                                                                          ExoArmsRecordingTraits::recording_channels.end()));
}

LiveExoArmsTrackerImpl::LiveExoArmsTrackerImpl(const OpenXRSessionHandles& handles,
                                               const ExoArmsTracker* tracker,
                                               std::unique_ptr<ExoArmsMcapChannels> mcap_channels)
    : mcap_channels_(std::move(mcap_channels)),
      m_schema_reader(handles,
                      make_exo_arms_tensor_config(tracker),
                      mcap_channels_.get(),
                      /*mcap_channel_index=*/0,
                      /*mcap_channel_tracked_index=*/1)
{
}

void LiveExoArmsTrackerImpl::update(int64_t /*monotonic_time_ns*/)
{
    // Policy: SchemaTracker throws on critical OpenXR/tensor API failures.
    // Missing collection/no new data are treated as common non-fatal cases.
    m_schema_reader.update(m_tracked.data);
}

const ExoArmsOutputTrackedT& LiveExoArmsTrackerImpl::get_data() const
{
    return m_tracked;
}

} // namespace core
