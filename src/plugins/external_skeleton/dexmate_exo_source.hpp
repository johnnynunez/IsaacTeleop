// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "external_skeleton_source.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace external_skeleton
{

/*!
 * @brief Receives Dexmate exoskeleton joint state from the local Python bridge
 *        (``dexmate_bridge.py``) and surfaces it on the
 *        ``IExternalSkeletonSource`` contract.
 *
 * The bridge runs as a separate Python process that subscribes to dexmate's
 * ``dexcomm`` topic (default: ``/exo_joints``, see
 * ``omniteleop/leader/arm_reader.py``) and forwards each ``ExoJointData``
 * sample as a small fixed-format binary UDP packet to ``127.0.0.1:<port>``.
 *
 * Wire format (all little-endian; matches ``dexmate_bridge.py`` exactly):
 *
 *   offset  size  field
 *   ------  ----  -----------------------------------------------------------
 *      0     4    magic       = 0x4F584544 ('DEXO' little-endian)
 *      4     1    version     = 1
 *      5     1    has_velocity (0 or 1)
 *      6     1    n_left      (joint count for left arm,  0..32)
 *      7     1    n_right     (joint count for right arm, 0..32)
 *      8     8    timestamp_ns (int64, ExoJointData.timestamp_ns)
 *     16   n_left * 4   left_pos  (float32, radians)
 *     ...   n_left * 4   left_vel  (float32, rad/s, only if has_velocity)
 *     ...   n_right * 4  right_pos (float32, radians)
 *     ...   n_right * 4  right_vel (float32, rad/s, only if has_velocity)
 *
 * The listener socket is non-blocking; ``poll`` drains all queued packets and
 * keeps the latest. Returns ``false`` until at least one packet has arrived
 * (so consumers see ``data == null`` rather than zero-filled state).
 */
class DexmateExoSource : public IExternalSkeletonSource
{
public:
    //! Default UDP port that ``dexmate_bridge.py`` binds to.
    static constexpr uint16_t DEFAULT_PORT = 53700;
    //! Per-arm joint cap; rejects oversized packets defensively.
    static constexpr uint8_t MAX_JOINTS_PER_ARM = 32;

    /*!
     * @param bind_host Interface to bind on (e.g. ``"127.0.0.1"`` for a
     *                  local-only bridge).
     * @param bind_port UDP port the bridge will send to.
     */
    DexmateExoSource(const std::string& bind_host, uint16_t bind_port);
    ~DexmateExoSource() override;

    DexmateExoSource(const DexmateExoSource&) = delete;
    DexmateExoSource& operator=(const DexmateExoSource&) = delete;

    std::string source_id() const override
    {
        return "dexmate-vega";
    }

    bool poll(core::ExternalSkeletonPoseT& out, int64_t& raw_device_clock_ns) override;

private:
    int socket_fd_ = -1;
    std::string bind_host_;
    uint16_t bind_port_;

    // Reusable receive buffer; sized for the max packet:
    // 16-byte header + 2 arms × 32 joints × 2 fields × 4 bytes = 528 bytes.
    std::vector<uint8_t> recv_buffer_;
};

} // namespace external_skeleton
} // namespace plugins
