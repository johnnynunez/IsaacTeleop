// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Plays an H.264 Annex B file into a Televiz QuadLayer:
//   ./viz_video_smoke /path/to/video.h264
// The example loops the stream on EOF.

#include "h264_file_reader.hpp"
#include "nvdec_player.hpp"

#include <viz/core/vk_context.hpp>
#include <viz/layers/quad_layer.hpp>
#include <viz/session/viz_session.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

namespace
{

// Pull NAL units out of the demuxer until NvDecoder produces at
// least one queued frame. Returns false only if the file is empty.
bool prime_first_frame(viz_smoke::H264FileReader& reader, viz_smoke::NvdecPlayer& player)
{
    for (int safety = 0; safety < 4096; ++safety)
    {
        const uint8_t* nalu = nullptr;
        size_t nalu_size = 0;
        if (!reader.next_nalu(&nalu, &nalu_size))
        {
            return false;
        }
        if (player.feed(nalu, nalu_size))
        {
            return true;
        }
    }
    return false;
}

void submit_to_layer(viz::QuadLayer& layer, const viz_smoke::OwnedRgba& f)
{
    viz::VizBuffer src{};
    src.data = f.data;
    src.width = f.width;
    src.height = f.height;
    src.format = viz::PixelFormat::kRGBA8;
    src.pitch = static_cast<size_t>(f.width) * 4;
    src.space = viz::MemorySpace::kDevice;
    layer.submit(src);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "usage: %s <video.h264>\n"
                     "  Input must be raw H.264 Annex B. To convert from MP4:\n"
                     "    ffmpeg -i in.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 out.h264\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    try
    {
        viz_smoke::H264FileReader reader(argv[1]);
        viz_smoke::NvdecPlayer player;

        if (!prime_first_frame(reader, player))
        {
            throw std::runtime_error("video_smoke: never produced a decoded frame; bad input?");
        }
        auto first = player.try_pop();

        viz::VizSession::Config cfg{};
        cfg.mode = viz::DisplayMode::kWindow;
        cfg.window_width = first->width;
        cfg.window_height = first->height;
        cfg.app_name = "viz_video_smoke";
        cfg.clear_color[0] = 0.0f;
        cfg.clear_color[1] = 0.0f;
        cfg.clear_color[2] = 0.0f;
        cfg.clear_color[3] = 1.0f;

        auto session = viz::VizSession::create(cfg);
        const viz::VkContext* ctx = session->get_vk_context();
        const VkRenderPass render_pass = session->get_render_pass();

        viz::QuadLayer::Config layer_cfg;
        layer_cfg.name = "video";
        layer_cfg.resolution = { first->width, first->height };
        auto* layer = session->add_layer<viz::QuadLayer>(*ctx, render_pass, layer_cfg);

        submit_to_layer(*layer, *first);

        // Holds the buffer most recently submitted. QuadLayer::submit
        // schedules an async cudaMemcpy from this buffer into the
        // mailbox slot; the renderer waits on cuda_done_writing
        // before sampling. By keeping the buffer alive until we
        // submit a different one (after a full render cycle), the
        // memcpy is guaranteed to have completed before cudaFree
        // (in ~OwnedRgba) runs.
        std::unique_ptr<viz_smoke::OwnedRgba> in_flight = std::move(first);

        while (!session->should_close())
        {
            // Top up the decoder until it has at least one queued
            // frame OR we run out of bitstream. Bound the inner loop
            // so a malformed file can't spin us forever.
            for (int safety = 0; safety < 256 && player.queued_frame_count() == 0; ++safety)
            {
                const uint8_t* nalu = nullptr;
                size_t nalu_size = 0;
                if (!reader.next_nalu(&nalu, &nalu_size))
                {
                    break;
                }
                player.feed(nalu, nalu_size);
            }

            if (auto next = player.try_pop())
            {
                submit_to_layer(*layer, *next);
                // Drop the previous buffer here. Its memcpy completed
                // during the last session->render() — the mailbox's
                // cuda_done_writing wait + the trailing fence wait
                // together guarantee no GPU work still references it.
                in_flight = std::move(next);
            }

            const auto info = session->render();
            if (info.frame_index > 0 && info.frame_index % 60 == 0)
            {
                const auto stats = session->get_frame_timing_stats();
                std::printf("frame %llu: %.1f fps (%.2f ms/frame, decoded queue=%zu)\n",
                            static_cast<unsigned long long>(info.frame_index), stats.render_fps,
                            stats.avg_frame_time_ms, player.queued_frame_count());
                std::fflush(stdout);
            }
        }

        session.reset();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "viz_video_smoke: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
