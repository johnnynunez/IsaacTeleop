// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <viz/session/display_backend.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace viz
{

class GlfwWindow;
class Swapchain;

// kWindow backend: GLFW window + Vulkan swapchain. Layers render
// into an intermediate RT; record_post_render_pass blits intermediate
// → swapchain image with the right layout transitions; end_frame
// presents.
class WindowBackend final : public DisplayBackend
{
public:
    struct Config
    {
        uint32_t width = 1024;
        uint32_t height = 1024;
        std::string title = "televiz";
        // Soft fps cap. 0 = use the primary monitor's refresh rate
        // (queried via GLFW at init). With MAILBOX present mode the
        // WSI doesn't throttle us, so without a cap we'd burn the GPU
        // at thousands of fps. Set to a positive value to override
        // (useful for benchmarks).
        uint32_t target_fps = 0;
    };

    explicit WindowBackend(Config config);
    ~WindowBackend() override;

    std::vector<std::string> required_instance_extensions() const override;
    std::vector<std::string> required_device_extensions() const override;
    void init(const VkContext& ctx, Resolution preferred_size) override;

    std::optional<Frame> begin_frame(int64_t predicted_display_time) override;
    const RenderTarget& render_target() const override;
    void record_post_render_pass(VkCommandBuffer cmd, const Frame& frame) override;
    void end_frame(const Frame& frame) override;

    void poll_events() override;
    bool should_close() const override;
    bool consume_resized() override;
    void resize(Resolution new_size) override;
    Resolution current_extent() const override;

    void destroy();

private:
    Config config_;
    const VkContext* ctx_ = nullptr;

    std::unique_ptr<GlfwWindow> window_;
    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<RenderTarget> render_target_;

    // Frame pacing. With MAILBOX present mode, the WSI never blocks
    // our acquire; on a fast GPU we'd run at thousands of fps and
    // peg power. The pacer runs at the START of begin_frame (before
    // acquire) so it always executes once per render iteration —
    // even when begin_frame returns nullopt (OUT_OF_DATE recovery).
    // Putting it at end_frame would skip pacing on early returns
    // and produce tight spin loops. Period is queried from the
    // primary monitor's GLFW video mode at init.
    std::chrono::nanoseconds frame_period_{ 0 };
    std::chrono::steady_clock::time_point next_frame_deadline_{};

    // Per-frame: image_index from the most recent begin_frame() ride
    // out through end_frame() via Frame::backend_token. Stored as
    // uint64_t there; cast back here.
    static constexpr uint64_t kNoImage = UINT64_MAX;
};

} // namespace viz
