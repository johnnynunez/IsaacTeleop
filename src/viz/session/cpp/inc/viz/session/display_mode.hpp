// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace viz
{

// Display backend for a VizSession. Lives in its own header so
// VizSession::Config and VizCompositor::Config can both reference it
// without including each other (VizSession owns VizCompositor).
//
// kOffscreen renders to an internal framebuffer with readback support
// (CI / tests). kWindow opens a GLFW window and presents via a Vulkan
// swapchain. kXr ships with the OpenXR backend.
enum class DisplayMode
{
    kOffscreen,
    kWindow,
    kXr,
};

} // namespace viz
