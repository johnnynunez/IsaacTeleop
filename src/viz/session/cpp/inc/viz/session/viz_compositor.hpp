// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <viz/core/frame_sync.hpp>
#include <viz/core/host_image.hpp>
#include <viz/core/viz_types.hpp>
#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

namespace viz
{

class DisplayBackend;
class LayerBase;
class VkContext;

// VizCompositor: per-session GPU pipeline that runs one render pass
// per frame. Drives a non-owning DisplayBackend for everything mode-
// specific (target image, present, readback). Owns the per-frame
// fence and the command pool / buffer.
//
// Lifetime: owned by VizSession. Created when the session moves from
// kUninitialized to kReady (after the backend has been created and
// initialized); destroyed when the session is destroyed.
class VizCompositor
{
public:
    struct Config
    {
        VkClearColorValue clear_color{ { 0.0f, 0.0f, 0.0f, 1.0f } };
    };

    static std::unique_ptr<VizCompositor> create(const VkContext& ctx, DisplayBackend& backend, const Config& config);

    ~VizCompositor();
    void destroy();

    VizCompositor(const VizCompositor&) = delete;
    VizCompositor& operator=(const VizCompositor&) = delete;
    VizCompositor(VizCompositor&&) = delete;
    VizCompositor& operator=(VizCompositor&&) = delete;

    // Records and submits one frame.
    //   1. backend.begin_frame() -> Frame (or skip).
    //   2. Snapshot visible layers; compute per-layer tile rects from
    //      their aspect_ratio() hints.
    //   3. Begin render pass on backend.render_target(); pre-bind
    //      scissor per layer (tile.outer); call layer->record() with
    //      per-layer ViewInfo (viewport = tile.content).
    //   4. End render pass; backend.record_post_render_pass() does
    //      any blit / transition the backend needs.
    //   5. Submit, waiting on layers' cuda_done_writing +
    //      frame.wait_before_render, signaling frame.signal_after_render.
    //   6. backend.end_frame() — present / xrEndFrame / no-op.
    //   7. fence wait — synchronous frame (mailbox layers depend on
    //      this — see quad_layer.hpp).
    void render(const std::vector<LayerBase*>& layers);

    // Forwards to backend; convenience for VizSession.
    HostImage readback_to_host();

    VkRenderPass render_pass() const noexcept;
    Resolution resolution() const noexcept;

private:
    VizCompositor(const VkContext& ctx, DisplayBackend& backend, const Config& config);
    void init();

    void create_command_pool();
    void create_command_buffer();

    // vkQueueSubmit wrapper that recovers the fence if submit fails.
    // After frame_sync_->reset(), the fence is unsignaled; if the real
    // submit then fails, the next frame_sync_->wait() would deadlock
    // forever on UINT64_MAX. On submit failure we attempt an empty
    // no-op submit so the fence gets signaled, converting "silent
    // hang" into "throw on next call" — the caller can then destroy +
    // recreate the session.
    void submit_or_signal_fence(const VkSubmitInfo& info, const char* what);

    const VkContext* ctx_ = nullptr;
    DisplayBackend* backend_ = nullptr;
    Config config_{};

    std::unique_ptr<FrameSync> frame_sync_;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
};

} // namespace viz
