// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <viz/session/display_backend.hpp>

#include <memory>

namespace viz
{

// kOffscreen backend: layers render into an intermediate RenderTarget
// and the result is read back to host memory on demand. No present,
// no events. Used by tests and by callers that consume frames as
// numpy/host arrays (CI, debug tooling).
class OffscreenBackend final : public DisplayBackend
{
public:
    OffscreenBackend();
    ~OffscreenBackend() override;

    void init(const VkContext& ctx, Resolution preferred_size) override;

    std::optional<Frame> begin_frame(int64_t predicted_display_time) override;
    const RenderTarget& render_target() const override;

    Resolution current_extent() const override;

    // Allocates a tightly-packed RGBA8 host buffer and copies the
    // intermediate RT's color attachment into it. Synchronous.
    HostImage readback_to_host() override;

    void destroy();

private:
    void create_readback_staging();
    void destroy_readback_staging();

    const VkContext* ctx_ = nullptr;
    Resolution extent_{};
    std::unique_ptr<RenderTarget> render_target_;

    // Pre-allocated host-visible staging buffer reused per readback.
    VkBuffer readback_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory readback_memory_ = VK_NULL_HANDLE;
    VkDeviceSize readback_byte_size_ = 0;

    // Per-call command pool/buffer for the readback copy. Separate
    // from the compositor's command buffer so readback never races
    // the per-frame command buffer recording.
    VkCommandPool readback_command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer readback_command_buffer_ = VK_NULL_HANDLE;
};

} // namespace viz
