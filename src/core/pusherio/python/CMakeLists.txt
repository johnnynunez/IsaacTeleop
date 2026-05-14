# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pybind11_add_module(pusherio_py
    pusherio_bindings.cpp
)

target_link_libraries(pusherio_py
    PRIVATE
        pusherio::pusherio
)

set_target_properties(pusherio_py PROPERTIES
    OUTPUT_NAME "_pusherio"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_package/$<CONFIG>/isaacteleop/pusherio"
)
