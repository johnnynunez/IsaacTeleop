// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dexmate_exo_source.hpp"
#include "external_skeleton_plugin.hpp"
#include "synthetic_skeleton_source.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

using namespace plugins::external_skeleton;

namespace
{

uint16_t parse_port(const char* s)
{
    const long v = std::strtol(s, nullptr, 10);
    if (v <= 0 || v > 65535)
    {
        throw std::invalid_argument(std::string("ExternalSkeletonPlugin: invalid port '") + s + "'");
    }
    return static_cast<uint16_t>(v);
}

// TODO(<vendor>): When integrating an additional device, add a new
// IExternalSkeletonSource subclass and a new branch here. The plugin core
// (ExternalSkeletonPlugin) is intentionally agnostic to which source is used.
std::unique_ptr<IExternalSkeletonSource> make_source(const std::string& source_kind, int argc, char** argv)
{
    if (source_kind == "synthetic")
    {
        return std::make_unique<SyntheticSkeletonSource>();
    }
    if (source_kind == "dexmate")
    {
        // Optional positional args: <bind_host> <bind_port>
        const std::string bind_host = (argc > 3) ? argv[3] : "127.0.0.1";
        const uint16_t bind_port = (argc > 4) ? parse_port(argv[4]) : DexmateExoSource::DEFAULT_PORT;
        return std::make_unique<DexmateExoSource>(bind_host, bind_port);
    }
    throw std::invalid_argument("ExternalSkeletonPlugin: unknown source kind '" + source_kind +
                                "' (supported: synthetic, dexmate)");
}

} // namespace

int main(int argc, char** argv)
try
{
    const std::string source_kind = (argc > 1) ? argv[1] : "synthetic";
    const std::string collection_id = (argc > 2) ? argv[2] : "external_skeleton";

    std::cout << "ExternalSkeletonPlugin (source: " << source_kind << ", collection: " << collection_id << ")"
              << std::endl;

    ExternalSkeletonPlugin plugin(make_source(source_kind, argc, argv), collection_id);

    // Push at 60 Hz upper bound; sources poll at their own native rate
    // (Dexmate's arm_reader publishes at ~40 Hz by default — extra ticks here
    // are no-ops since DexmateExoSource::poll returns false when nothing new
    // has arrived).
    // TODO: Make this rate configurable per source.
    const auto frame_duration = std::chrono::nanoseconds(1000000000 / 60);
    const auto program_start = std::chrono::steady_clock::now();
    std::size_t frame_count = 0;

    while (true)
    {
        plugin.update();
        frame_count++;
        std::this_thread::sleep_until(program_start + frame_duration * frame_count);
    }

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << argv[0] << ": " << e.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << argv[0] << ": Unknown error" << std::endl;
    return 1;
}
