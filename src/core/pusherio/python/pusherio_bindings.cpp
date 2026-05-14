// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the PusherIO C++ library. Exposes SchemaPusher and
// SchemaPusherConfig so that Python plugin processes can push serialized
// FlatBuffer data via the OpenXR runtime alongside the existing C++ plugins.

#include <pusherio/schema_pusher.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

PYBIND11_MODULE(_pusherio, m)
{
    // The schema bindings already pull in oxr session types as part of importing
    // isaacteleop.oxr, but ensure the converters for OpenXRSessionHandles are
    // available even when a caller only imports isaacteleop.pusherio.
    py::module_::import("isaacteleop.oxr._oxr");

    m.doc() = "Isaac Teleop PusherIO - SchemaPusher for pushing serialized FlatBuffer data via OpenXR";

    py::class_<core::SchemaPusherConfig>(m, "SchemaPusherConfig")
        .def(py::init<>())
        .def(py::init(
                 [](std::string collection_id, std::size_t max_flatbuffer_size, std::string tensor_identifier,
                    std::string localized_name, std::string app_name)
                 {
                     core::SchemaPusherConfig cfg;
                     cfg.collection_id = std::move(collection_id);
                     cfg.max_flatbuffer_size = max_flatbuffer_size;
                     cfg.tensor_identifier = std::move(tensor_identifier);
                     cfg.localized_name = std::move(localized_name);
                     cfg.app_name = std::move(app_name);
                     return cfg;
                 }),
             py::arg("collection_id"), py::arg("max_flatbuffer_size"), py::arg("tensor_identifier"),
             py::arg("localized_name"), py::arg("app_name") = std::string{})
        .def_readwrite("collection_id", &core::SchemaPusherConfig::collection_id)
        .def_readwrite("max_flatbuffer_size", &core::SchemaPusherConfig::max_flatbuffer_size)
        .def_readwrite("tensor_identifier", &core::SchemaPusherConfig::tensor_identifier)
        .def_readwrite("localized_name", &core::SchemaPusherConfig::localized_name)
        .def_readwrite("app_name", &core::SchemaPusherConfig::app_name);

    py::class_<core::SchemaPusher>(m, "SchemaPusher")
        .def_static("get_required_extensions", &core::SchemaPusher::get_required_extensions,
                    "Return the OpenXR extension names required to construct a SchemaPusher")
        .def(py::init<const core::OpenXRSessionHandles&, core::SchemaPusherConfig>(), py::arg("handles"),
             py::arg("config"),
             "Construct a SchemaPusher bound to an existing OpenXRSession (created with the required extensions)")
        .def(
            "push_buffer",
            [](core::SchemaPusher& self, py::buffer buffer, std::int64_t sample_time_local_common_clock_ns,
               std::int64_t sample_time_raw_device_clock_ns)
            {
                py::buffer_info info = buffer.request();
                if (info.ndim != 1)
                {
                    throw std::invalid_argument("SchemaPusher.push_buffer: buffer must be 1-D");
                }
                if (info.itemsize != 1)
                {
                    throw std::invalid_argument(
                        "SchemaPusher.push_buffer: buffer must be bytes-like (itemsize == 1)");
                }
                const auto size = static_cast<std::size_t>(info.size);
                const auto* ptr = static_cast<const std::uint8_t*>(info.ptr);
                self.push_buffer(ptr, size, sample_time_local_common_clock_ns, sample_time_raw_device_clock_ns);
            },
            py::arg("buffer"), py::arg("sample_time_local_common_clock_ns"),
            py::arg("sample_time_raw_device_clock_ns"),
            "Push a serialized FlatBuffer payload over the OpenXR tensor collection. The buffer "
            "must be a bytes-like 1-D object (bytes, bytearray, memoryview, numpy uint8 array, etc.).")
        .def("config", &core::SchemaPusher::config, py::return_value_policy::reference_internal,
             "Get the SchemaPusherConfig used to construct this SchemaPusher");
}
