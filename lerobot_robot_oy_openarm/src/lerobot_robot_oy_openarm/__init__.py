#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .config_oy_openarm_follower import (
    OyOpenArmFollowerConfig,
    OyOpenArmFollowerConfigBase,
)
from .foxglove_async import install_async_foxglove_logging
from .oy_openarm_follower import OY_OPENARM_POSITION_FEATURES, OyOpenArmFollower

# ``lerobot-record`` imports its visualization functions before discovering
# third-party plugins.  Installing here lets the standard command keep
# Foxglove work out of the 30 Hz control loop without modifying LeRobot itself.
install_async_foxglove_logging()

__all__ = [
    "OY_OPENARM_POSITION_FEATURES",
    "OyOpenArmFollower",
    "OyOpenArmFollowerConfig",
    "OyOpenArmFollowerConfigBase",
    "install_async_foxglove_logging",
]
