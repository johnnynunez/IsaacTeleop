// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace plugins
{
namespace manus
{

// Vendor binding for the Teleop -> Manus haptic-glove tensor collection.
// The Teleop-side producer is a generic
// isaacteleop.haptic_devices.push_tensor.PushTensorHapticDevice (see the
// isaacteleop.haptic_devices.glove.haptic_glove_device factory and the
// haptic_feedback example). Whatever consumer the app wires must pass this
// same collection_id string so the runtime pairs them by name.
inline constexpr const char* MANUS_GLOVE_COLLECTION_ID = "manus_glove_haptic";

// A HapticCommand carries an endpoint string ("left"/"right") plus a
// 5-finger values vector; serialised it is ~64 B. 128 leaves headroom for
// the longer endpoint names and FlatBuffer padding without bloating
// per-collection storage.
inline constexpr std::size_t MANUS_GLOVE_MAX_PAYLOAD_SIZE = 128;

} // namespace manus
} // namespace plugins
