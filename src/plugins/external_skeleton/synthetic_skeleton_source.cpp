// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "synthetic_skeleton_source.hpp"

#include <oxr_utils/os_time.hpp>

#include <cmath>

namespace plugins
{
namespace external_skeleton
{

namespace
{

constexpr float kPi = 3.14159265358979323846f;

// Sized to match Dexmate Vega's per-arm DOF (see PUBLISH_THRESHOLD in
// omniteleop's leader/arm_reader.py: 7 joints per arm). Easy to grow if
// you point the synthetic source at a different downstream config.
constexpr uint32_t kJointsPerArm = 7;

void fill_arm(core::ExoArmJointStateT& arm, float t, float phase)
{
    arm.positions_radians.resize(kJointsPerArm);
    arm.velocities_radians_per_sec.resize(kJointsPerArm);
    for (uint32_t j = 0; j < kJointsPerArm; ++j)
    {
        // Offset each joint by a small phase so the wave is visible across
        // the chain rather than every joint moving identically.
        const float w = 2.0f * kPi * 0.5f; // 0.5 Hz fundamental
        const float p = phase + 0.25f * static_cast<float>(j);
        arm.positions_radians[j] = 0.5f * std::sin(w * t + p);
        arm.velocities_radians_per_sec[j] = 0.5f * w * std::cos(w * t + p);
    }
}

} // namespace

SyntheticSkeletonSource::SyntheticSkeletonSource() : start_(std::chrono::steady_clock::now())
{
}

bool SyntheticSkeletonSource::poll(core::ExternalSkeletonPoseT& out, int64_t& raw_device_clock_ns)
{
    const auto now = std::chrono::steady_clock::now();
    const float t = std::chrono::duration<float>(now - start_).count();

    if (!out.left_arm)
    {
        out.left_arm = std::make_shared<core::ExoArmJointStateT>();
    }
    if (!out.right_arm)
    {
        out.right_arm = std::make_shared<core::ExoArmJointStateT>();
    }
    fill_arm(*out.left_arm, t, /*phase=*/0.0f);
    fill_arm(*out.right_arm, t, /*phase=*/kPi); // anti-phase so left/right are visually distinguishable

    out.source_id = source_id();
    // Synthetic source has no device clock; surface the local monotonic clock
    // so downstream consumers still see a strictly monotonic device timestamp.
    out.device_timestamp_ns = core::os_monotonic_now_ns();
    raw_device_clock_ns = out.device_timestamp_ns;
    return true;
}

} // namespace external_skeleton
} // namespace plugins
