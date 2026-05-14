# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The exoskeleton plugin is a Python script (uses isaacteleop.pusherio,
# isaacteleop.schema, and the third-party ``dynamixel-sdk``), so there is
# nothing to compile. We install the script, manifest, and README alongside the
# C++ plugins.
#
# At runtime the user is expected to have the ``isaacteleop`` wheel installed
# (which provides the ``isaacteleop.pusherio.SchemaPusher`` and
# ``isaacteleop.schema.ExoArmsOutput`` Python bindings used by the script) and
# to have installed ``dynamixel-sdk`` (via ``pip install isaacteleop[exoskeleton]``).

install(PROGRAMS exoskeleton_plugin.py DESTINATION plugins/exoskeleton)
install(FILES plugin.yaml README.md DESTINATION plugins/exoskeleton)
