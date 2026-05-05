// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <viz/core/vk_context.hpp>
#include <viz/layers/layer_base.hpp>
#include <viz/session/swapchain.hpp>
#include <viz/session/tile_layout.hpp>
#include <viz/session/viz_compositor.hpp>

#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace viz
{

namespace
{

void check_vk(VkResult result, const char* what)
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("VizCompositor: ") + what + " failed: VkResult=" + std::to_string(result));
    }
}

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
    {
        if ((type_bits & (1u << i)) != 0 && (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("VizCompositor: no memory type matches readback requirements");
}

} // namespace

std::unique_ptr<VizCompositor> VizCompositor::create(const VkContext& ctx, const Config& config)
{
    if (!ctx.is_initialized())
    {
        throw std::invalid_argument("VizCompositor: VkContext is not initialized");
    }
    if (config.resolution.width == 0 || config.resolution.height == 0)
    {
        throw std::invalid_argument("VizCompositor: resolution must be non-zero");
    }
    if (config.mode == DisplayMode::kWindow && config.swapchain == nullptr)
    {
        throw std::invalid_argument("VizCompositor: kWindow requires a non-null swapchain");
    }
    std::unique_ptr<VizCompositor> c(new VizCompositor(ctx, config));
    c->init();
    return c;
}

VizCompositor::VizCompositor(const VkContext& ctx, const Config& config) : ctx_(&ctx), config_(config)
{
}

VizCompositor::~VizCompositor()
{
    destroy();
}

void VizCompositor::init()
{
    try
    {
        render_target_ = RenderTarget::create(*ctx_, RenderTarget::Config{ config_.resolution });
        frame_sync_ = FrameSync::create(*ctx_);
        create_command_pool();
        create_command_buffer();
        // Readback staging is only useful in kOffscreen — kWindow / kXr
        // present via swapchain and don't expose host readback.
        if (config_.mode == DisplayMode::kOffscreen)
        {
            create_readback_staging();
        }
    }
    catch (...)
    {
        destroy();
        throw;
    }
}

void VizCompositor::handle_resize(Resolution new_size)
{
    if (config_.mode != DisplayMode::kWindow || config_.swapchain == nullptr)
    {
        return;
    }
    if (new_size.width == 0 || new_size.height == 0)
    {
        // GLFW reports (0, 0) when the window is minimized; defer the
        // recreate until the user un-minimizes (next non-zero size).
        return;
    }
    // Drain GPU work before tearing down the intermediate RT — frame
    // commands may still be in flight if the previous frame was the
    // one that observed the resize.
    (void)vkDeviceWaitIdle(ctx_->device());

    config_.swapchain->recreate(new_size);
    config_.resolution = config_.swapchain->extent();

    // Rebuild the intermediate RT at the new size. Render pass remains
    // valid (its compatibility doesn't depend on extent), but the
    // VkImage / VkImageView / VkFramebuffer must be recreated.
    render_target_ = RenderTarget::create(*ctx_, RenderTarget::Config{ config_.resolution });
}

void VizCompositor::destroy()
{
    if (ctx_ == nullptr)
    {
        return;
    }
    const VkDevice device = ctx_->device();
    if (device == VK_NULL_HANDLE)
    {
        return;
    }
    if (readback_buffer_ != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, readback_buffer_, nullptr);
        readback_buffer_ = VK_NULL_HANDLE;
    }
    if (readback_memory_ != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, readback_memory_, nullptr);
        readback_memory_ = VK_NULL_HANDLE;
    }
    readback_byte_size_ = 0;
    if (command_pool_ != VK_NULL_HANDLE)
    {
        // Pool destruction frees all command buffers allocated from it.
        vkDestroyCommandPool(device, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
        command_buffer_ = VK_NULL_HANDLE;
    }
    frame_sync_.reset();
    render_target_.reset();
}

void VizCompositor::create_command_pool()
{
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = ctx_->queue_family_index();
    check_vk(vkCreateCommandPool(ctx_->device(), &info, nullptr, &command_pool_), "vkCreateCommandPool");
}

void VizCompositor::create_command_buffer()
{
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = command_pool_;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    check_vk(vkAllocateCommandBuffers(ctx_->device(), &info, &command_buffer_), "vkAllocateCommandBuffers");
}

void VizCompositor::create_readback_staging()
{
    // Sized to one tightly-packed RGBA8 frame at the configured
    // resolution. destroy() owns cleanup; readback_to_host() never
    // allocates per call.
    readback_byte_size_ = static_cast<VkDeviceSize>(config_.resolution.width) * config_.resolution.height *
                          bytes_per_pixel(PixelFormat::kRGBA8);

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = readback_byte_size_;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check_vk(vkCreateBuffer(ctx_->device(), &bi, nullptr, &readback_buffer_), "vkCreateBuffer(readback staging)");

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx_->device(), readback_buffer_, &reqs);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = find_memory_type(ctx_->physical_device(), reqs.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check_vk(vkAllocateMemory(ctx_->device(), &ai, nullptr, &readback_memory_), "vkAllocateMemory(readback staging)");
    check_vk(vkBindBufferMemory(ctx_->device(), readback_buffer_, readback_memory_, 0),
             "vkBindBufferMemory(readback staging)");
}

void VizCompositor::submit_or_signal_fence(const VkSubmitInfo& info, const char* what)
{
    const VkResult r = vkQueueSubmit(ctx_->queue(), 1, &info, frame_sync_->in_flight_fence());
    if (r == VK_SUCCESS)
    {
        return;
    }
    // Real submit failed; the fence is still unsignaled. Best-effort
    // signal it via an empty no-op submit so the next wait() throws
    // (or returns) instead of deadlocking on UINT64_MAX. If this also
    // fails the original error still propagates and the caller should
    // destroy + recreate the session.
    VkSubmitInfo empty{};
    empty.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    (void)vkQueueSubmit(ctx_->queue(), 1, &empty, frame_sync_->in_flight_fence());
    throw std::runtime_error(std::string("VizCompositor: ") + what + " failed: VkResult=" + std::to_string(r));
}

namespace
{

Rect2D to_rect2d(const VkRect2D& r)
{
    return Rect2D{ r.offset.x, r.offset.y, r.extent.width, r.extent.height };
}

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
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

} // namespace

void VizCompositor::render(const std::vector<LayerBase*>& layers, const std::vector<ViewInfo>& views)
{
    // Wait for the previous frame's GPU work to complete before reusing
    // the command buffer / fence (1 frame in flight today).
    frame_sync_->wait();

    // Snapshot the visible-layer set ONCE per frame. is_visible() is
    // an atomic flag; sampling it twice across record / wait-collect
    // would let a mid-frame toggle record draws but skip the
    // matching cuda_done_writing wait (or vice versa), which would
    // race the producer's CUDA copy.
    std::vector<LayerBase*> visible_layers;
    visible_layers.reserve(layers.size());
    for (LayerBase* layer : layers)
    {
        if (layer != nullptr && layer->is_visible())
        {
            visible_layers.push_back(layer);
        }
    }

    // kWindow: acquire the next swapchain image. Out-of-date or
    // suboptimal returns nullopt; we drop this frame and let the
    // session call handle_resize() before the next render(). Returning
    // here leaves frame_sync_ signaled from the previous wait(), so
    // the next render() doesn't deadlock.
    std::optional<Swapchain::AcquiredImage> acquired;
    if (config_.mode == DisplayMode::kWindow)
    {
        acquired = config_.swapchain->acquire_next_image();
        if (!acquired.has_value())
        {
            return;
        }
    }

    // Build per-layer tile rects (kWindow only). For each visible
    // layer the tile_layout helper returns:
    //   outer:    the equal-slice tile (used as the layer's scissor —
    //             confines all draws to this layer's region).
    //   content:  the aspect-fit rect inside outer (used as the
    //             layer's per-view viewport — letterbox margins keep
    //             the framebuffer's clear color).
    std::vector<TileSlot> tiles;
    if (config_.mode == DisplayMode::kWindow && !visible_layers.empty())
    {
        const float fb_aspect =
            static_cast<float>(config_.resolution.width) / static_cast<float>(config_.resolution.height);
        std::vector<float> aspects;
        aspects.reserve(visible_layers.size());
        for (LayerBase* layer : visible_layers)
        {
            // Layers without a preferred aspect fill their full tile.
            aspects.push_back(layer->aspect_ratio().value_or(fb_aspect));
        }
        tiles = tile_layout(aspects, config_.resolution, /*padding=*/0);
    }

    check_vk(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command_buffer_, &begin), "vkBeginCommandBuffer");

    std::array<VkClearValue, 2> clears{};
    clears[0].color = config_.clear_color;
    clears[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = render_target_->render_pass();
    rp.framebuffer = render_target_->framebuffer();
    rp.renderArea.offset = { 0, 0 };
    rp.renderArea.extent = { config_.resolution.width, config_.resolution.height };
    rp.clearValueCount = static_cast<uint32_t>(clears.size());
    rp.pClearValues = clears.data();

    vkCmdBeginRenderPass(command_buffer_, &rp, VK_SUBPASS_CONTENTS_INLINE);

    // Per-layer dispatch. Pre-bind scissor (= tile.outer in window,
    // full-fb in offscreen) so any draw that escapes the layer's
    // viewport is clipped. Build per-layer ViewInfo with viewport
    // overridden to tile.content (or full-fb in offscreen).
    const VkRect2D full_fb_rect{ { 0, 0 }, { config_.resolution.width, config_.resolution.height } };
    for (size_t i = 0; i < visible_layers.size(); ++i)
    {
        const VkRect2D scissor_rect = (config_.mode == DisplayMode::kWindow) ? tiles[i].outer : full_fb_rect;
        const VkRect2D viewport_rect = (config_.mode == DisplayMode::kWindow) ? tiles[i].content : full_fb_rect;
        vkCmdSetScissor(command_buffer_, 0, 1, &scissor_rect);

        // Per-layer copy of `views` with the viewport rect overridden.
        // In window/offscreen views.size() == 1; in XR == 2 (per-eye
        // viewports come from the OpenXR runtime, not from the tile).
        std::vector<ViewInfo> layer_views(views.begin(), views.end());
        if (layer_views.empty())
        {
            layer_views.push_back(ViewInfo{});
        }
        layer_views[0].viewport = to_rect2d(viewport_rect);
        visible_layers[i]->record(command_buffer_, layer_views, *render_target_);
    }

    vkCmdEndRenderPass(command_buffer_);

    // kWindow: blit the intermediate framebuffer to the swapchain
    // image, transition for present.
    if (acquired.has_value())
    {
        transition_image(command_buffer_, acquired->image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        const VkExtent2D sc_extent = { config_.swapchain->extent().width, config_.swapchain->extent().height };
        VkImageBlit region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.srcOffsets[1] = { static_cast<int32_t>(config_.resolution.width),
                                 static_cast<int32_t>(config_.resolution.height), 1 };
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.layerCount = 1;
        region.dstOffsets[1] = { static_cast<int32_t>(sc_extent.width), static_cast<int32_t>(sc_extent.height), 1 };
        vkCmdBlitImage(command_buffer_, render_target_->color_image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       acquired->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

        transition_image(command_buffer_, acquired->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    check_vk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    // Reset the fence immediately before submit. If anything between
    // wait() and here threw (a layer's record(), a Vulkan API failure
    // during recording), the fence stays signaled from the previous
    // frame and the next render() doesn't deadlock on wait().
    frame_sync_->reset();

    // Collect layer-provided wait timeline semaphores + (in window mode)
    // the swapchain's image-available semaphore. Flatten into the
    // arrays vkQueueSubmit expects, with a chained
    // VkTimelineSemaphoreSubmitInfo for the per-semaphore counter
    // values (ignored on binary semaphores; padded with 0).
    std::vector<VkSemaphore> wait_semaphores;
    std::vector<uint64_t> wait_values;
    std::vector<VkPipelineStageFlags> wait_stages;
    for (LayerBase* layer : visible_layers)
    {
        for (const auto& w : layer->get_wait_semaphores())
        {
            if (w.semaphore != VK_NULL_HANDLE)
            {
                wait_semaphores.push_back(w.semaphore);
                wait_values.push_back(w.value);
                wait_stages.push_back(w.wait_stage);
            }
        }
    }
    if (acquired.has_value())
    {
        wait_semaphores.push_back(acquired->image_available);
        wait_values.push_back(0); // binary semaphore — value ignored
        wait_stages.push_back(VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    std::vector<VkSemaphore> signal_semaphores;
    std::vector<uint64_t> signal_values;
    if (acquired.has_value())
    {
        signal_semaphores.push_back(acquired->render_done);
        signal_values.push_back(0); // binary semaphore — value ignored
    }

    VkTimelineSemaphoreSubmitInfo timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
    timeline.pWaitSemaphoreValues = wait_values.empty() ? nullptr : wait_values.data();
    timeline.signalSemaphoreValueCount = static_cast<uint32_t>(signal_values.size());
    timeline.pSignalSemaphoreValues = signal_values.empty() ? nullptr : signal_values.data();

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &timeline;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    submit.waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size());
    submit.pWaitSemaphores = wait_semaphores.empty() ? nullptr : wait_semaphores.data();
    submit.pWaitDstStageMask = wait_stages.empty() ? nullptr : wait_stages.data();
    submit.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size());
    submit.pSignalSemaphores = signal_semaphores.empty() ? nullptr : signal_semaphores.data();
    submit_or_signal_fence(submit, "vkQueueSubmit");

    // kWindow: queue the present (waits on render_done). Out-of-date
    // returns false; we still drain via frame_sync_->wait() below so
    // the next handle_resize() call sees idle GPU state.
    if (acquired.has_value())
    {
        (void)config_.swapchain->present(acquired->image_index, acquired->render_done);
    }

    // Wait for completion before returning so readback / next frame sees
    // a consistent state. With 1 frame in flight this is the natural
    // synchronization point; multi-buffered swapchain rendering moves
    // this wait to the start of the next frame. QuadLayer's mailbox
    // depends on this — see quad_layer.hpp.
    frame_sync_->wait();
}

HostImage VizCompositor::readback_to_host()
{
    // Reuses the staging buffer allocated at init() — no per-call alloc,
    // no cleanup-on-throw concerns. Buffer lifetime tracks the
    // compositor's; destroy() frees it.
    const uint32_t w = config_.resolution.width;
    const uint32_t h = config_.resolution.height;

    // Record + submit a single copy. The render pass already transitioned
    // the color image to TRANSFER_SRC_OPTIMAL, so no barrier is needed.
    check_vk(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer(readback)");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command_buffer_, &begin), "vkBeginCommandBuffer(readback)");

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { w, h, 1 };
    vkCmdCopyImageToBuffer(command_buffer_, render_target_->color_image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback_buffer_, 1, &region);

    check_vk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer(readback)");

    frame_sync_->reset();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    submit_or_signal_fence(submit, "vkQueueSubmit(readback)");
    frame_sync_->wait();

    HostImage result(config_.resolution, PixelFormat::kRGBA8);
    void* mapped = nullptr;
    check_vk(vkMapMemory(ctx_->device(), readback_memory_, 0, readback_byte_size_, 0, &mapped), "vkMapMemory(readback)");
    std::memcpy(result.data(), mapped, readback_byte_size_);
    vkUnmapMemory(ctx_->device(), readback_memory_);

    return result;
}

VkRenderPass VizCompositor::render_pass() const noexcept
{
    return render_target_ ? render_target_->render_pass() : VK_NULL_HANDLE;
}

Resolution VizCompositor::resolution() const noexcept
{
    return config_.resolution;
}

} // namespace viz
