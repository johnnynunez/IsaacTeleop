// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/exo_arms_tracker_base.hpp>
#include <schema/exo_arms_generated.h>

#include <cstddef>
#include <string>

namespace core
{

/*!
 * @brief Facade for dual-arm exoskeleton joint state exposed as ``ExoArmsOutputTrackedT``.
 *
 * Semantic contract: ``left_arm_pos``, ``left_arm_vel``, ``right_arm_pos``, ``right_arm_vel`` are
 * variable-length float vectors carrying joint positions (radians) and velocities (rad/s) for a
 * dual-arm exoskeleton (e.g. the Dexmate Vega exoskeleton, 7 DOF per arm). Vector sizes are
 * defined by the data producer; pos and vel vectors for a given arm must agree in size, and
 * either vel vector may be empty if the producer was configured without velocity reading.
 * Units, joint ordering, and calibration are set by the upstream producer (e.g. the
 * IsaacTeleop ``exoskeleton`` plugin: ``src/plugins/exoskeleton/exoskeleton_plugin.py``).
 *
 * After each ``ITrackerSession::update()`` that includes this tracker, ``get_data(session)``
 * reflects the implementation's tracked snapshot. The live backend retains the last-known
 * sample when a tick has no new samples (collection still present) and clears ``data`` to null
 * when the OpenXR tensor collection becomes unavailable.
 *
 * Usage:
 * @code
 * auto tracker = std::make_shared<ExoArmsTracker>("my_exo_collection");
 * // ... register the tracker with a session, then each tick: ...
 * session->update();
 * const auto& data = tracker->get_data(*session);
 * if (data.data) {
 *     // data.data->left_arm_pos, etc.
 * }
 * @endcode
 */
class ExoArmsTracker : public ITracker
{
public:
    //! Default maximum FlatBuffer size for ExoArmsOutput messages.
    //! 4 vectors * ~16 floats max * 4 bytes + FlatBuffer overhead.
    //! Generous default to accommodate up to ~64 joints total without rebuild.
    static constexpr size_t DEFAULT_MAX_FLATBUFFER_SIZE = 1024;

    /*!
     * @brief Constructs an ExoArmsTracker.
     * @param collection_id Logical stream identifier; must match the exo plugin's collection_id.
     * @param max_flatbuffer_size Upper bound for serialized ``ExoArmsOutput`` / record payloads
     *        (default: 1024 bytes); must accommodate the joint count produced by the plugin.
     */
    explicit ExoArmsTracker(const std::string& collection_id,
                            size_t max_flatbuffer_size = DEFAULT_MAX_FLATBUFFER_SIZE);

    std::string_view get_name() const override
    {
        return TRACKER_NAME;
    }

    /*!
     * @brief Exo arms snapshot from the session's implementation.
     *
     * ``tracked.data`` is null until the first valid sample arrives or after the collection
     * disappears. When non-null, the nested ``left_arm_pos`` / ``right_arm_pos`` etc. vectors
     * are safe to read; their sizes are producer-defined (typically 7 per arm for the Vega exo).
     */
    const ExoArmsOutputTrackedT& get_data(const ITrackerSession& session) const;

    const std::string& collection_id() const
    {
        return collection_id_;
    }

    size_t max_flatbuffer_size() const
    {
        return max_flatbuffer_size_;
    }

private:
    static constexpr const char* TRACKER_NAME = "ExoArmsTracker";

    std::string collection_id_;
    size_t max_flatbuffer_size_;
};

} // namespace core
