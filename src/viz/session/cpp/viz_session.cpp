// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <viz/session/glfw_window.hpp>
#include <viz/session/swapchain.hpp>
#include <viz/session/viz_session.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <stdexcept>

namespace viz
{

namespace
{

void reject_xr(DisplayMode mode, const char* what)
{
    if (mode == DisplayMode::kXr)
    {
        throw std::runtime_error(std::string("VizSession: ") + what +
                                 " is not implemented for kXr (XR backend ships in M5)");
    }
}

std::vector<std::string> glfw_required_instance_extensions_or_throw()
{
    uint32_t count = 0;
    const char** raw = glfwGetRequiredInstanceExtensions(&count);
    if (raw == nullptr)
    {
        throw std::runtime_error(
            "VizSession: glfwGetRequiredInstanceExtensions returned null "
            "(no Vulkan loader visible to GLFW)");
    }
    std::vector<std::string> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        out.emplace_back(raw[i]);
    }
    return out;
}

} // namespace

std::unique_ptr<VizSession> VizSession::create(const Config& config)
{
    if (config.window_width == 0 || config.window_height == 0)
    {
        throw std::invalid_argument("VizSession: window dimensions must be non-zero");
    }
    std::unique_ptr<VizSession> s(new VizSession(config));
    s->init();
    return s;
}

VizSession::VizSession(const Config& config) : config_(config)
{
}

VizSession::~VizSession()
{
    destroy();
}

void VizSession::init()
{
    // kXr is the only mode not implemented yet; kOffscreen + kWindow
    // ship now. Reject early to avoid a wasted vkCreateInstance on a
    // mode we can't support.
    reject_xr(config_.mode, "create");

    try
    {
        // Build the VkContext config based on display mode. kWindow
        // needs GLFW's required instance extensions + VK_KHR_swapchain.
        VkContext::Config vk_cfg{};
        if (config_.mode == DisplayMode::kWindow)
        {
            vk_cfg.instance_extensions = glfw_required_instance_extensions_or_throw();
            vk_cfg.device_extensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }

        // Acquire / create the Vulkan context.
        if (config_.external_context != nullptr)
        {
            if (!config_.external_context->is_initialized())
            {
                throw std::invalid_argument("VizSession: external_context is not initialized");
            }
            ctx_ptr_ = config_.external_context;
        }
        else
        {
            owned_ctx_ = std::make_unique<VkContext>();
            owned_ctx_->init(vk_cfg);
            ctx_ptr_ = owned_ctx_.get();
        }

        // For kWindow: open the GLFW window + Vulkan swapchain. The
        // intermediate render target's resolution matches the swapchain
        // extent so the post-render blit is 1:1.
        Resolution render_res{ config_.window_width, config_.window_height };
        if (config_.mode == DisplayMode::kWindow)
        {
            window_ = GlfwWindow::create(ctx_ptr_->instance(), config_.window_width, config_.window_height,
                                         config_.app_name);
            swapchain_ = Swapchain::create(*ctx_ptr_, window_->surface(),
                                           Resolution{ config_.window_width, config_.window_height });
            render_res = swapchain_->extent();
        }

        VizCompositor::Config c_cfg{};
        c_cfg.resolution = render_res;
        c_cfg.clear_color = { { config_.clear_color[0], config_.clear_color[1], config_.clear_color[2],
                                config_.clear_color[3] } };
        c_cfg.mode = config_.mode;
        c_cfg.swapchain = swapchain_.get();
        compositor_ = VizCompositor::create(*ctx_ptr_, c_cfg);

        state_ = SessionState::kReady;
    }
    catch (...)
    {
        destroy();
        throw;
    }
}

void VizSession::destroy()
{
    layers_.clear();
    compositor_.reset();
    // Order: swapchain holds VkSurfaceKHR refs (drains on destroy);
    // window owns the surface; both must outlive the device but be
    // destroyed before the VkContext.
    swapchain_.reset();
    window_.reset();
    if (owned_ctx_)
    {
        owned_ctx_.reset();
    }
    ctx_ptr_ = nullptr;
    state_ = SessionState::kDestroyed;
}

void VizSession::remove_layer(LayerBase* layer)
{
    if (layer == nullptr)
    {
        return;
    }
    auto it = std::remove_if(
        layers_.begin(), layers_.end(), [layer](const std::unique_ptr<LayerBase>& p) { return p.get() == layer; });
    layers_.erase(it, layers_.end());
}

FrameInfo VizSession::begin_frame()
{
    if (state_ == SessionState::kDestroyed || state_ == SessionState::kLost)
    {
        throw std::runtime_error("VizSession: begin_frame called on destroyed/lost session");
    }
    if (frame_in_progress_)
    {
        throw std::logic_error(
            "VizSession: begin_frame called while a frame is already in "
            "progress (missing end_frame for previous begin_frame)");
    }
    if (state_ == SessionState::kReady)
    {
        state_ = SessionState::kRunning;
    }

    const auto now = std::chrono::steady_clock::now();
    if (first_frame_)
    {
        current_frame_info_.delta_time = 0.0f;
        first_frame_ = false;
    }
    else
    {
        current_frame_info_.delta_time = std::chrono::duration<float>(now - last_frame_time_).count();
    }
    last_frame_time_ = now;

    current_frame_info_.frame_index = frame_index_;
    current_frame_info_.predicted_display_time = 0; // XR-only; 0 in offscreen
    current_frame_info_.should_render = (state_ == SessionState::kRunning);
    current_frame_info_.resolution = compositor_->resolution();
    // Single identity view in window/offscreen; XR backend extends to per-eye.
    current_frame_info_.views.assign(1, ViewInfo{});

    // Set last so any earlier throw leaves the flag false and the next
    // begin_frame() can proceed normally.
    frame_in_progress_ = true;

    return current_frame_info_;
}

void VizSession::end_frame()
{
    if (!frame_in_progress_)
    {
        throw std::logic_error("VizSession: end_frame called without a matching begin_frame");
    }
    if (state_ != SessionState::kRunning)
    {
        // No-op in non-running states (matches the design: kStopping
        // submits an empty frame; kReady never enters end_frame).
        // Still clear the in-progress flag so the pairing contract holds.
        frame_in_progress_ = false;
        return;
    }

    // Always clear the in-progress flag, even if the render call below
    // throws — leaving it true would lock out all subsequent begin_frame()
    // calls for the rest of the session.
    struct ClearGuard
    {
        bool* flag;
        ~ClearGuard()
        {
            *flag = false;
        }
    } guard{ &frame_in_progress_ };

    // Build a raw-pointer view of the layer registry for the compositor —
    // avoids forcing the compositor to know about std::unique_ptr.
    std::vector<LayerBase*> raw_layers;
    raw_layers.reserve(layers_.size());
    for (const auto& l : layers_)
    {
        raw_layers.push_back(l.get());
    }

    if (current_frame_info_.should_render)
    {
        compositor_->render(raw_layers, current_frame_info_.views);
    }

    update_timing_stats(current_frame_info_.delta_time);
    ++frame_index_;
}

FrameInfo VizSession::render()
{
    if (window_)
    {
        // Pump GLFW events first — drives close button, resize callback,
        // any input handlers users register on the window.
        window_->poll_events();
        if (window_->consume_resized())
        {
            // Defer to compositor: drain device, recreate swapchain +
            // intermediate RT at the new framebuffer size.
            compositor_->handle_resize(window_->framebuffer_size());
        }
    }
    auto info = begin_frame();
    end_frame();
    return info;
}

void VizSession::update_timing_stats(float frame_time_seconds)
{
    if (frame_time_seconds <= 0.0f)
    {
        return;
    }
    // Simple exponential moving average; full FPS smoothing arrives with
    // the window/XR backends' real frame pacing.
    constexpr float kSmoothing = 0.1f;
    const float frame_ms = frame_time_seconds * 1000.0f;
    timing_stats_.avg_frame_time_ms = kSmoothing * frame_ms + (1.0f - kSmoothing) * timing_stats_.avg_frame_time_ms;
    timing_stats_.render_fps =
        (timing_stats_.avg_frame_time_ms > 0.0f) ? 1000.0f / timing_stats_.avg_frame_time_ms : 0.0f;
}

Resolution VizSession::get_recommended_resolution() const noexcept
{
    return compositor_ ? compositor_->resolution() : Resolution{ config_.window_width, config_.window_height };
}

HostImage VizSession::readback_to_host()
{
    if (config_.mode != DisplayMode::kOffscreen)
    {
        throw std::runtime_error(
            "VizSession::readback_to_host: only kOffscreen supports host readback "
            "(use the swapchain present path in kWindow / kXr)");
    }
    if (!compositor_)
    {
        throw std::runtime_error("VizSession: readback_to_host called before init");
    }
    return compositor_->readback_to_host();
}

bool VizSession::should_close() const noexcept
{
    return window_ ? window_->should_close() : false;
}

const VkContext& VizSession::ctx() const noexcept
{
    return *ctx_ptr_;
}

VkDevice VizSession::get_vk_device() const noexcept
{
    return ctx_ptr_ ? ctx_ptr_->device() : VK_NULL_HANDLE;
}

VkPhysicalDevice VizSession::get_vk_physical_device() const noexcept
{
    return ctx_ptr_ ? ctx_ptr_->physical_device() : VK_NULL_HANDLE;
}

uint32_t VizSession::get_vk_queue_family_index() const noexcept
{
    return ctx_ptr_ ? ctx_ptr_->queue_family_index() : UINT32_MAX;
}

VkRenderPass VizSession::get_render_pass() const noexcept
{
    return compositor_ ? compositor_->render_pass() : VK_NULL_HANDLE;
}

const VkContext* VizSession::get_vk_context() const noexcept
{
    return ctx_ptr_;
}

} // namespace viz
