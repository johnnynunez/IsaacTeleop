// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dexmate_exo_source.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace plugins
{
namespace external_skeleton
{

namespace
{

// 'DEXO' as little-endian uint32 (D=0x44, E=0x45, X=0x58, O=0x4F).
constexpr uint32_t kMagic = 0x4F584544u;
constexpr uint8_t kVersion = 1;
constexpr size_t kHeaderSize = 16;

constexpr size_t kMaxPacketSize =
    kHeaderSize + 2u * static_cast<size_t>(DexmateExoSource::MAX_JOINTS_PER_ARM) * 2u * sizeof(float);

// Read a little-endian uint32 / uint64 / float32 without relying on host
// endianness. The bridge writes little-endian explicitly so this is the
// portable way to decode on any host.
uint32_t read_u32_le(const uint8_t* p)
{
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

int64_t read_i64_le(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
    {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return static_cast<int64_t>(v);
}

float read_f32_le(const uint8_t* p)
{
    const uint32_t u = read_u32_le(p);
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

} // namespace

DexmateExoSource::DexmateExoSource(const std::string& bind_host, uint16_t bind_port)
    : bind_host_(bind_host), bind_port_(bind_port), recv_buffer_(kMaxPacketSize)
{
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0)
    {
        throw std::runtime_error(std::string("DexmateExoSource: socket() failed: ") + std::strerror(errno));
    }

    int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags < 0 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw std::runtime_error(std::string("DexmateExoSource: fcntl(O_NONBLOCK) failed: ") + std::strerror(errno));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(bind_port_);
    if (inet_pton(AF_INET, bind_host_.c_str(), &addr.sin_addr) != 1)
    {
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw std::runtime_error("DexmateExoSource: invalid bind host '" + bind_host_ + "'");
    }

    if (::bind(socket_fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        const int err = errno;
        ::close(socket_fd_);
        socket_fd_ = -1;
        throw std::runtime_error("DexmateExoSource: bind(" + bind_host_ + ":" + std::to_string(bind_port_) +
                                 ") failed: " + std::strerror(err));
    }

    std::cout << "DexmateExoSource: listening for dexmate_bridge.py on " << bind_host_ << ":" << bind_port_ << std::endl;
}

DexmateExoSource::~DexmateExoSource()
{
    if (socket_fd_ >= 0)
    {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

bool DexmateExoSource::poll(core::ExternalSkeletonPoseT& out, int64_t& raw_device_clock_ns)
{
    bool got_any = false;
    ssize_t n = 0;
    while ((n = ::recv(socket_fd_, recv_buffer_.data(), recv_buffer_.size(), 0)) > 0)
    {
        // Drain the socket; the most recent packet wins (older packets in the
        // queue are stale by definition since the bridge publishes at ~40 Hz).
        if (static_cast<size_t>(n) < kHeaderSize)
        {
            std::cerr << "DexmateExoSource: ignoring short packet (" << n << " bytes)" << std::endl;
            continue;
        }

        const uint8_t* p = recv_buffer_.data();
        if (read_u32_le(p) != kMagic)
        {
            std::cerr << "DexmateExoSource: bad magic, ignoring packet" << std::endl;
            continue;
        }
        const uint8_t version = p[4];
        if (version != kVersion)
        {
            std::cerr << "DexmateExoSource: unsupported version " << static_cast<int>(version) << " (expected "
                      << static_cast<int>(kVersion) << ")" << std::endl;
            continue;
        }
        const bool has_vel = p[5] != 0;
        const uint8_t n_left = p[6];
        const uint8_t n_right = p[7];
        if (n_left > MAX_JOINTS_PER_ARM || n_right > MAX_JOINTS_PER_ARM)
        {
            std::cerr << "DexmateExoSource: joint count exceeds cap (left=" << static_cast<int>(n_left)
                      << ", right=" << static_cast<int>(n_right) << ", cap=" << static_cast<int>(MAX_JOINTS_PER_ARM)
                      << ")" << std::endl;
            continue;
        }

        const size_t per_arm_floats = (has_vel ? 2u : 1u) * static_cast<size_t>(n_left + n_right);
        const size_t expected_size = kHeaderSize + per_arm_floats * sizeof(float);
        if (static_cast<size_t>(n) != expected_size)
        {
            std::cerr << "DexmateExoSource: size mismatch (got " << n << ", expected " << expected_size << ")"
                      << std::endl;
            continue;
        }

        const int64_t device_ts_ns = read_i64_le(p + 8);

        if (!out.left_arm)
        {
            out.left_arm = std::make_shared<core::ExoArmJointStateT>();
        }
        if (!out.right_arm)
        {
            out.right_arm = std::make_shared<core::ExoArmJointStateT>();
        }
        out.left_arm->positions_radians.resize(n_left);
        out.right_arm->positions_radians.resize(n_right);
        out.left_arm->velocities_radians_per_sec.resize(has_vel ? n_left : 0);
        out.right_arm->velocities_radians_per_sec.resize(has_vel ? n_right : 0);

        const uint8_t* cursor = p + kHeaderSize;
        for (uint8_t i = 0; i < n_left; ++i, cursor += sizeof(float))
        {
            out.left_arm->positions_radians[i] = read_f32_le(cursor);
        }
        if (has_vel)
        {
            for (uint8_t i = 0; i < n_left; ++i, cursor += sizeof(float))
            {
                out.left_arm->velocities_radians_per_sec[i] = read_f32_le(cursor);
            }
        }
        for (uint8_t i = 0; i < n_right; ++i, cursor += sizeof(float))
        {
            out.right_arm->positions_radians[i] = read_f32_le(cursor);
        }
        if (has_vel)
        {
            for (uint8_t i = 0; i < n_right; ++i, cursor += sizeof(float))
            {
                out.right_arm->velocities_radians_per_sec[i] = read_f32_le(cursor);
            }
        }

        out.source_id = source_id();
        out.device_timestamp_ns = device_ts_ns;
        raw_device_clock_ns = device_ts_ns;
        got_any = true;
    }

    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        // Real error (not just "no data right now") — surface but don't kill
        // the plugin; the next tick may recover.
        std::cerr << "DexmateExoSource: recv error: " << std::strerror(errno) << std::endl;
    }

    return got_any;
}

} // namespace external_skeleton
} // namespace plugins
