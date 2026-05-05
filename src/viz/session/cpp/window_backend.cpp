// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <viz/core/vk_context.hpp>
#include <viz/session/glfw_window.hpp>
#include <viz/session/swapchain.hpp>
#include <viz/session/window_backend.hpp>

#include <stdexcept>
#include <utility>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace viz
{

namespace
{

void transition_image(VkCommandBuffer cmd,
                      VkImage image,
                      VkImageLayout old_layout,
                      VkImageLayout new_layout,
                      VkAccessFlags src_access,
                      VkAccessFlags dst_access,
                      VkPipelineStageFlags src_stage,
                      VkPipelineStageFlags dst_stage)
{
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

} // namespace

WindowBackend::WindowBackend(Config config) : config_(std::move(config))
{
}

WindowBackend::~WindowBackend()
{
    destroy();
}

std::vector<std::string> WindowBackend::required_instance_extensions() const
{
    // GLFW reports the surface extensions for the current platform
    // (VK_KHR_surface + the platform-specific one — xlib/wayland/win32).
    // glfwInit must succeed before this query; GlfwWindow::create()
    // refcounts init separately, but querying extensions doesn't
    // require a window.
    if (glfwInit() != GLFW_TRUE)
    {
        throw std::runtime_error(
            "WindowBackend: glfwInit failed — no display available "
            "for kWindow mode");
    }
    uint32_t count = 0;
    const char** raw = glfwGetRequiredInstanceExtensions(&count);
    if (raw == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error(
            "WindowBackend: glfwGetRequiredInstanceExtensions returned null "
            "(no Vulkan loader visible to GLFW)");
    }
    std::vector<std::string> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        out.emplace_back(raw[i]);
    }
    glfwTerminate();
    return out;
}

std::vector<std::string> WindowBackend::required_device_extensions() const
{
    return { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
}

void WindowBackend::init(const VkContext& ctx, Resolution preferred_size)
{
    ctx_ = &ctx;
    try
    {
        window_ = GlfwWindow::create(ctx.instance(), preferred_size.width, preferred_size.height, config_.title);
        swapchain_ = Swapchain::create(ctx, window_->surface(), preferred_size);
        // Match intermediate RT extent to the swapchain so the post-
        // render blit is 1:1.
        render_target_ = RenderTarget::create(ctx, RenderTarget::Config{ swapchain_->extent() });
    }
    catch (...)
    {
        destroy();
        throw;
    }
}

void WindowBackend::destroy()
{
    // Order matters: RT and swapchain hold device resources that must
    // be torn down before the window's surface, which itself must
    // outlive any swapchain ref. ctx is non-owning; leave alone.
    render_target_.reset();
    swapchain_.reset();
    window_.reset();
    ctx_ = nullptr;
}

std::optional<DisplayBackend::Frame> WindowBackend::begin_frame(int64_t /*predicted_display_time*/)
{
    if (swapchain_ == nullptr)
    {
        return std::nullopt;
    }
    auto acquired = swapchain_->acquire_next_image();
    if (!acquired.has_value())
    {
        // Out-of-date / suboptimal — caller (compositor / session)
        // will skip + recreate via consume_resized() on next frame.
        return std::nullopt;
    }

    Frame f{};
    f.views.assign(1, ViewInfo{});
    f.views[0].viewport = Rect2D{ 0, 0, swapchain_->extent().width, swapchain_->extent().height };
    f.wait_before_render = acquired->image_available;
    f.wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    f.signal_after_render = acquired->render_done;
    f.backend_token = static_cast<uint64_t>(acquired->image_index);
    // Stash the swapchain image too — record_post_render_pass needs
    // it. Pack into a higher-bit slot of backend_token's payload:
    // the AcquiredImage's `image` lives only as long as the swapchain
    // doesn't recreate, which it can't between begin and end_frame
    // (the trailing fence wait gates it). So we just look it up by
    // index in record_post_render_pass via a fresh acquire query.
    // Simpler: also stash the VkImage as a side cache on the backend.
    // (See pending_blit_image_ if added; for now we re-query by index.)
    return f;
}

const RenderTarget& WindowBackend::render_target() const
{
    if (render_target_ == nullptr)
    {
        throw std::runtime_error("WindowBackend::render_target: backend not initialized");
    }
    return *render_target_;
}

void WindowBackend::record_post_render_pass(VkCommandBuffer cmd, const Frame& frame)
{
    if (swapchain_ == nullptr || render_target_ == nullptr)
    {
        return;
    }
    const uint32_t image_index = static_cast<uint32_t>(frame.backend_token);
    // Look up the swapchain image directly — Swapchain doesn't
    // currently expose images_ by index, but we know the image_index
    // fits in [0, image_count). Add an accessor for clarity.
    // (Falls back to UNDEFINED layout transition if Swapchain
    // exposes nothing — bug; see Swapchain::image(uint32_t).)
    const VkImage swap_image = swapchain_->image_at(image_index);

    transition_image(cmd, swap_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT);

    const Resolution intermediate_extent{ render_target_->resolution() };
    const Resolution sc_extent = swapchain_->extent();
    VkImageBlit region{};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.srcOffsets[1] = { static_cast<int32_t>(intermediate_extent.width),
                             static_cast<int32_t>(intermediate_extent.height), 1 };
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.dstOffsets[1] = { static_cast<int32_t>(sc_extent.width), static_cast<int32_t>(sc_extent.height), 1 };
    vkCmdBlitImage(cmd, render_target_->color_image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swap_image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

    transition_image(cmd, swap_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void WindowBackend::end_frame(const Frame& frame)
{
    if (swapchain_ == nullptr)
    {
        return;
    }
    const uint32_t image_index = static_cast<uint32_t>(frame.backend_token);
    // Out-of-date / suboptimal returns false; we let the next frame's
    // begin_frame() observe it and the session catches it via
    // consume_resized() in the GLFW callback.
    (void)swapchain_->present(image_index, frame.signal_after_render);
}

void WindowBackend::poll_events()
{
    if (window_)
    {
        window_->poll_events();
    }
}

bool WindowBackend::should_close() const
{
    return window_ ? window_->should_close() : false;
}

bool WindowBackend::consume_resized()
{
    return window_ ? window_->consume_resized() : false;
}

void WindowBackend::resize(Resolution new_size)
{
    if (swapchain_ == nullptr || ctx_ == nullptr)
    {
        return;
    }
    if (new_size.width == 0 || new_size.height == 0)
    {
        // Window is minimized — defer until non-zero size.
        return;
    }
    (void)vkDeviceWaitIdle(ctx_->device());
    swapchain_->recreate(new_size);
    // RenderTarget recreation is cheap; the new render pass is
    // compatible with the prior so layer pipelines stay valid.
    render_target_ = RenderTarget::create(*ctx_, RenderTarget::Config{ swapchain_->extent() });
}

Resolution WindowBackend::current_extent() const
{
    if (swapchain_ != nullptr)
    {
        return swapchain_->extent();
    }
    return Resolution{ config_.width, config_.height };
}

} // namespace viz
