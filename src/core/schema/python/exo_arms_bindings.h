// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python bindings for the ExoArms FlatBuffer schema.
// Types: ExoArmsOutput, ExoArmsOutputTracked, ExoArmsOutputRecord.

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <schema/exo_arms_generated.h>
#include <schema/timestamp_generated.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace core
{

inline void bind_exo_arms(py::module& m)
{
    py::class_<ExoArmsOutputT, std::shared_ptr<ExoArmsOutputT>>(m, "ExoArmsOutput")
        .def(py::init([]() { return std::make_shared<ExoArmsOutputT>(); }))
        .def(py::init(
                 [](const std::vector<float>& left_arm_pos, const std::vector<float>& left_arm_vel,
                    const std::vector<float>& right_arm_pos, const std::vector<float>& right_arm_vel)
                 {
                     auto obj = std::make_shared<ExoArmsOutputT>();
                     obj->left_arm_pos = left_arm_pos;
                     obj->left_arm_vel = left_arm_vel;
                     obj->right_arm_pos = right_arm_pos;
                     obj->right_arm_vel = right_arm_vel;
                     return obj;
                 }),
             py::arg("left_arm_pos"), py::arg("left_arm_vel"), py::arg("right_arm_pos"), py::arg("right_arm_vel"))
        .def_property(
            "left_arm_pos", [](const ExoArmsOutputT& self) { return self.left_arm_pos; },
            [](ExoArmsOutputT& self, const std::vector<float>& v) { self.left_arm_pos = v; })
        .def_property(
            "left_arm_vel", [](const ExoArmsOutputT& self) { return self.left_arm_vel; },
            [](ExoArmsOutputT& self, const std::vector<float>& v) { self.left_arm_vel = v; })
        .def_property(
            "right_arm_pos", [](const ExoArmsOutputT& self) { return self.right_arm_pos; },
            [](ExoArmsOutputT& self, const std::vector<float>& v) { self.right_arm_pos = v; })
        .def_property(
            "right_arm_vel", [](const ExoArmsOutputT& self) { return self.right_arm_vel; },
            [](ExoArmsOutputT& self, const std::vector<float>& v) { self.right_arm_vel = v; })
        .def(
            "serialize",
            [](const ExoArmsOutputT& self, size_t initial_capacity)
            {
                flatbuffers::FlatBufferBuilder builder(initial_capacity);
                auto offset = ExoArmsOutput::Pack(builder, &self);
                builder.Finish(offset);
                return py::bytes(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                                 static_cast<size_t>(builder.GetSize()));
            },
            py::arg("initial_capacity") = 1024,
            "Serialize this ExoArmsOutput to a FlatBuffer-encoded ``bytes`` payload ready for "
            "SchemaPusher.push_buffer(). The optional initial_capacity hints the FlatBufferBuilder's "
            "starting buffer size in bytes; the buffer grows automatically if needed.")
        .def("__repr__",
             [](const ExoArmsOutputT& self)
             {
                 return "ExoArmsOutput(left_arm_pos=[" + std::to_string(self.left_arm_pos.size()) +
                        "], left_arm_vel=[" + std::to_string(self.left_arm_vel.size()) +
                        "], right_arm_pos=[" + std::to_string(self.right_arm_pos.size()) +
                        "], right_arm_vel=[" + std::to_string(self.right_arm_vel.size()) + "])";
             });

    py::class_<ExoArmsOutputRecordT, std::shared_ptr<ExoArmsOutputRecordT>>(m, "ExoArmsOutputRecord")
        .def(py::init<>())
        .def(py::init(
                 [](const ExoArmsOutputT& data, const DeviceDataTimestamp& timestamp)
                 {
                     auto obj = std::make_shared<ExoArmsOutputRecordT>();
                     obj->data = std::make_shared<ExoArmsOutputT>(data);
                     obj->timestamp = std::make_shared<core::DeviceDataTimestamp>(timestamp);
                     return obj;
                 }),
             py::arg("data"), py::arg("timestamp"))
        .def_property_readonly(
            "data", [](const ExoArmsOutputRecordT& self) -> std::shared_ptr<ExoArmsOutputT> { return self.data; })
        .def_readonly("timestamp", &ExoArmsOutputRecordT::timestamp)
        .def("__repr__",
             [](const ExoArmsOutputRecordT& self) {
                 return "ExoArmsOutputRecord(data=" + std::string(self.data ? "ExoArmsOutput(...)" : "None") + ")";
             });

    py::class_<ExoArmsOutputTrackedT, std::shared_ptr<ExoArmsOutputTrackedT>>(m, "ExoArmsOutputTrackedT")
        .def(py::init<>())
        .def(py::init(
                 [](const ExoArmsOutputT& data)
                 {
                     auto obj = std::make_shared<ExoArmsOutputTrackedT>();
                     obj->data = std::make_shared<ExoArmsOutputT>(data);
                     return obj;
                 }),
             py::arg("data"))
        .def_property_readonly(
            "data", [](const ExoArmsOutputTrackedT& self) -> std::shared_ptr<ExoArmsOutputT> { return self.data; })
        .def("__repr__",
             [](const ExoArmsOutputTrackedT& self) {
                 return std::string("ExoArmsOutputTrackedT(data=") +
                        (self.data ? "ExoArmsOutput(...)" : "None") + ")";
             });
}

} // namespace core
