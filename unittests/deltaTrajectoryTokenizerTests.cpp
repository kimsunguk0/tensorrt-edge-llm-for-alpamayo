/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/deltaTrajectoryTokenizer.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace trt_edgellm;

namespace
{

traj::DeltaTrajectoryTokenizer::Mat3 makeYawRotation(float yaw)
{
    float const c = std::cos(yaw);
    float const s = std::sin(yaw);
    return {{{c, s, 0.0F}, {-s, c, 0.0F}, {0.0F, 0.0F, 1.0F}}};
}

} // namespace

TEST(DeltaTrajectoryTokenizerTest, EncodeHistoryXYZOnly)
{
    traj::DeltaTrajectoryTokenizer::Config config;
    config.egoXYZMin = {-1.0F, -1.0F, -1.0F};
    config.egoXYZMax = {1.0F, 1.0F, 1.0F};
    config.numBins = 10;

    traj::DeltaTrajectoryTokenizer tokenizer(config);
    std::vector<traj::DeltaTrajectoryTokenizer::Vec3> const histXYZ{
        {0.2F, -0.4F, 0.1F}, {0.6F, -0.8F, -0.2F}};

    std::vector<int32_t> const tokens = tokenizer.encodeHistory(histXYZ, {});
    std::vector<int32_t> const expected{5, 3, 5, 6, 3, 3};
    EXPECT_EQ(tokens, expected);
}

TEST(DeltaTrajectoryTokenizerTest, EncodeHistoryWithYaw)
{
    traj::DeltaTrajectoryTokenizer::Config config;
    config.egoXYZMin = {-1.0F, -1.0F, -1.0F};
    config.egoXYZMax = {1.0F, 1.0F, 1.0F};
    config.egoYawMin = -1.0F;
    config.egoYawMax = 1.0F;
    config.numBins = 10;
    config.predictYaw = true;

    traj::DeltaTrajectoryTokenizer tokenizer(config);
    std::vector<traj::DeltaTrajectoryTokenizer::Vec3> const histXYZ{
        {0.2F, -0.4F, 0.1F}, {0.6F, -0.8F, -0.2F}};
    std::vector<traj::DeltaTrajectoryTokenizer::Mat3> const histRot{
        makeYawRotation(0.2F), makeYawRotation(0.6F)};

    std::vector<int32_t> const tokens = tokenizer.encodeHistory(histXYZ, histRot);
    std::vector<int32_t> const expected{5, 3, 5, 5, 6, 3, 3, 6};
    EXPECT_EQ(tokens, expected);
}

TEST(DeltaTrajectoryTokenizerTest, FormatTokens)
{
    traj::DeltaTrajectoryTokenizer tokenizer;
    std::string const formatted = tokenizer.formatTokens({0, 5, 12}, 3000);
    EXPECT_EQ(formatted, "<|traj_history_start|><i3000><i3005><i3012><|traj_history_end|>");
}
