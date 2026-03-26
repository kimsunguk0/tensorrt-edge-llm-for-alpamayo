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

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace traj
{

class DeltaTrajectoryTokenizer
{
public:
    using Vec3 = std::array<float, 3>;
    using Mat3 = std::array<std::array<float, 3>, 3>;

    struct Config
    {
        std::array<float, 3> egoXYZMin{-4.0F, -4.0F, -10.0F};
        std::array<float, 3> egoXYZMax{4.0F, 4.0F, 10.0F};
        float egoYawMin{-3.14159265358979323846F};
        float egoYawMax{3.14159265358979323846F};
        int32_t numBins{1000};
        bool predictYaw{false};
    };

    DeltaTrajectoryTokenizer();
    explicit DeltaTrajectoryTokenizer(Config config);

    int32_t getVocabSize() const noexcept
    {
        return mConfig.numBins;
    }

    Config getConfig() const noexcept
    {
        return mConfig;
    }

    std::vector<int32_t> encode(std::vector<Vec3> const& histXYZ, std::vector<Mat3> const& histRot,
        std::vector<Vec3> const& futXYZ, std::vector<Mat3> const& futRot) const;

    std::vector<int32_t> encodeHistory(std::vector<Vec3> const& histXYZ, std::vector<Mat3> const& histRot) const;

    std::string formatTokens(std::vector<int32_t> const& tokens, int32_t tokenOffset = 0,
        std::string const& startToken = "<|traj_history_start|>",
        std::string const& endToken = "<|traj_history_end|>") const;

private:
    Config mConfig;

    int32_t quantizeXYZ(float value, int32_t dim) const;
    int32_t quantizeYaw(float value) const;
    static float extractYaw(Mat3 const& rotation);
    static float wrapAngleToPi(float angle);
};

} // namespace traj
} // namespace trt_edgellm
