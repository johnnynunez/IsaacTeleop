// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <viz/core/host_image.hpp>
#include <viz/core/render_target.hpp>
#include <viz/core/viz_types.hpp>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace viz
{

class VkContext;

// Abstract presentation target. VizSession instantiates one per
// DisplayMode; VizCompositor drives it.
//
// Each backend owns:
//   - The intermediate RenderTarget layers render into. Render-pass
//     handle stays compatible across resize so layer pipelines aren't
//     invalidated.
//   - Mode-specific resources: GLFW window + VkSwapchainKHR (kWindow),
//     readback staging buffer (kOffscreen), OpenXR session + XR
//     swapchains (kXr — M5).
//
// Per-frame contract:
//   1. VizCompositor calls begin_frame() — backend acquires anything
//      it needs (Vulkan swapchain image, XR predicted display time).
//      nullopt = "skip this frame" (out-of-date / shouldRender=false).
//   2. VizCompositor records the render pass into render_target().
//   3. VizCompositor calls record_post_render_pass() — backend issues
//      any cmds needed before submit (blit intermediate → swapchain
//      image + barriers in kWindow; no-op in kOffscreen).
//   4. VizCompositor builds vkSubmitInfo with the backend's wait/
//      signal semaphores plus the layers' cuda_done_writing waits,
//      submits.
//   5. VizCompositor calls end_frame() on submit success — backend
//      presents (kWindow) / xrEndFrame (kXr) / no-op (kOffscreen).
class DisplayBackend
{
public:
    virtual ~DisplayBackend() = default;

    DisplayBackend(const DisplayBackend&) = delete;
    DisplayBackend& operator=(const DisplayBackend&) = delete;
    DisplayBackend(DisplayBackend&&) = delete;
    DisplayBackend& operator=(DisplayBackend&&) = delete;

    // ------------------------------------------------------------
    // Setup phase. VizSession calls these in order:
    //   1. required_*_extensions() to populate VkContext::Config.
    //   2. init() once VkContext is up.
    // ------------------------------------------------------------

    // Vulkan instance/device extensions this backend needs. Empty by
    // default (kOffscreen). VizSession unions these into the
    // VkContext::Config before VkContext::init.
    virtual std::vector<std::string> required_instance_extensions() const
    {
        return {};
    }
    virtual std::vector<std::string> required_device_extensions() const
    {
        return {};
    }

    // Allocate device resources (intermediate RT + mode-specific
    // swapchain etc.). Throws on failure.
    virtual void init(const VkContext& ctx, Resolution preferred_size) = 0;

    // ------------------------------------------------------------
    // Per-frame phase. VizCompositor::render() calls these around
    // the render pass.
    // ------------------------------------------------------------

    struct Frame
    {
        // Per-view info. 1 entry in offscreen/window (full extent,
        // identity matrices); 2 in XR stereo (per-eye pose+fov+
        // viewport rect from xrLocateViews). VizCompositor overrides
        // viewport rects per-layer via tile_layout in window mode.
        std::vector<ViewInfo> views;

        // Wait/signal binary semaphores for the compositor's submit.
        // The compositor adds layer-side waits (cuda_done_writing) on
        // top of wait_before_render. VK_NULL_HANDLE = no semaphore
        // needed (kOffscreen).
        VkSemaphore wait_before_render = VK_NULL_HANDLE;
        VkPipelineStageFlags wait_stage = 0;
        VkSemaphore signal_after_render = VK_NULL_HANDLE;

        // Backend-private bookkeeping round-tripped to record_post_*
        // and end_frame (e.g. swapchain image_index in kWindow).
        uint64_t backend_token = 0;
    };

    // Acquires the next frame target. nullopt = skip this frame.
    virtual std::optional<Frame> begin_frame(int64_t predicted_display_time) = 0;

    // The intermediate RT layers render into. Same handle across the
    // backend's lifetime in offscreen/window; recreated by resize().
    // The RT's render pass is stable-compatible across recreate so
    // layer pipelines built against an earlier handle stay valid.
    virtual const RenderTarget& render_target() const = 0;

    // Record any cmds the backend needs after the layer render pass
    // and before vkEndCommandBuffer. Default: no-op (kOffscreen).
    virtual void record_post_render_pass(VkCommandBuffer /*cmd*/, const Frame& /*frame*/)
    {
    }

    // Called after the compositor's vkQueueSubmit succeeds (and after
    // the trailing fence wait, so the GPU is idle). Default: no-op.
    virtual void end_frame(const Frame& /*frame*/)
    {
    }

    // ------------------------------------------------------------
    // Lifecycle / event polling.
    // ------------------------------------------------------------

    // Pump platform events. kWindow drives GLFW here; the rest no-op.
    virtual void poll_events()
    {
    }

    // True iff the user / runtime has requested the session close.
    virtual bool should_close() const
    {
        return false;
    }

    // True iff a resize has been requested since the last consume.
    // Atomic-style read-and-clear. VizSession checks this at frame
    // start and calls resize() when set.
    virtual bool consume_resized()
    {
        return false;
    }

    // Drain device, tear down per-extent resources, recreate at the
    // new size. The render pass survives (stable-compatible).
    virtual void resize(Resolution /*new_size*/)
    {
    }

    // Current target extent. Drives the compositor's tile_layout +
    // viewport math.
    virtual Resolution current_extent() const = 0;

    // ------------------------------------------------------------
    // Optional: host-readback. Only kOffscreen overrides; the rest
    // throw because their target is a swapchain image / XR swapchain.
    // ------------------------------------------------------------

    virtual HostImage readback_to_host()
    {
        throw std::runtime_error("DisplayBackend: readback_to_host not supported on this backend");
    }

protected:
    DisplayBackend() = default;
};

} // namespace viz
