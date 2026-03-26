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

#include "common/checkMacros.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace traj
{

namespace
{

void validateConfig(DeltaTrajectoryTokenizer::Config const& config)
{
    check::check(config.numBins > 0, "DeltaTrajectoryTokenizer: numBins must be positive.");
    for (int32_t dim = 0; dim < 3; ++dim)
    {
        check::check(config.egoXYZMax[dim] > config.egoXYZMin[dim],
            "DeltaTrajectoryTokenizer: egoXYZMax must be greater than egoXYZMin for every dimension.");
    }
    check::check(config.egoYawMax > config.egoYawMin,
        "DeltaTrajectoryTokenizer: egoYawMax must be greater than egoYawMin.");
}

} // namespace

DeltaTrajectoryTokenizer::DeltaTrajectoryTokenizer()
    : DeltaTrajectoryTokenizer(Config{})
{
}

DeltaTrajectoryTokenizer::DeltaTrajectoryTokenizer(Config config)
    : mConfig(config)
{
    validateConfig(mConfig);
}

std::vector<int32_t> DeltaTrajectoryTokenizer::encode(std::vector<Vec3> const& histXYZ,
    std::vector<Mat3> const& histRot, std::vector<Vec3> const& futXYZ, std::vector<Mat3> const& futRot) const
{
    (void) histXYZ;
    (void) histRot;

    check::check(!futXYZ.empty(), "DeltaTrajectoryTokenizer::encode(): futXYZ must not be empty.");
    if (mConfig.predictYaw)
    {
        check::check(futRot.size() == futXYZ.size(),
            "DeltaTrajectoryTokenizer::encode(): futRot size must match futXYZ size when predictYaw is enabled.");
    }

    int32_t const featuresPerStep = mConfig.predictYaw ? 4 : 3;
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(futXYZ.size()) * featuresPerStep);

    Vec3 previousXYZ{0.0F, 0.0F, 0.0F};
    float previousYaw = 0.0F;

    for (size_t i = 0; i < futXYZ.size(); ++i)
    {
        Vec3 const deltaXYZ{
            futXYZ[i][0] - previousXYZ[0], futXYZ[i][1] - previousXYZ[1], futXYZ[i][2] - previousXYZ[2]};
        previousXYZ = futXYZ[i];

        tokens.push_back(quantizeXYZ(deltaXYZ[0], 0));
        tokens.push_back(quantizeXYZ(deltaXYZ[1], 1));
        tokens.push_back(quantizeXYZ(deltaXYZ[2], 2));

        if (mConfig.predictYaw)
        {
            float const yaw = extractYaw(futRot[i]);
            float const deltaYaw = wrapAngleToPi(yaw - previousYaw);
            previousYaw = yaw;
            tokens.push_back(quantizeYaw(deltaYaw));
        }
    }

    return tokens;
}

std::vector<int32_t> DeltaTrajectoryTokenizer::encodeHistory(
    std::vector<Vec3> const& histXYZ, std::vector<Mat3> const& histRot) const
{
    return encode(histXYZ, histRot, histXYZ, histRot);
}

std::string DeltaTrajectoryTokenizer::formatTokens(std::vector<int32_t> const& tokens, int32_t tokenOffset,
    std::string const& startToken, std::string const& endToken) const
{
    std::ostringstream ss;
    ss << startToken;
    for (int32_t token : tokens)
    {
        ss << "<i" << (tokenOffset + token) << ">";
    }
    ss << endToken;
    return ss.str();
}

int32_t DeltaTrajectoryTokenizer::quantizeXYZ(float value, int32_t dim) const
{
    float const normalized = (value - mConfig.egoXYZMin[dim]) / (mConfig.egoXYZMax[dim] - mConfig.egoXYZMin[dim]);
    float const scaled = normalized * static_cast<float>(mConfig.numBins - 1);
    long const rounded = std::lround(scaled);
    return static_cast<int32_t>(std::clamp<long>(rounded, 0, static_cast<long>(mConfig.numBins - 1)));
}

int32_t DeltaTrajectoryTokenizer::quantizeYaw(float value) const
{
    float const normalized = (value - mConfig.egoYawMin) / (mConfig.egoYawMax - mConfig.egoYawMin);
    float const scaled = normalized * static_cast<float>(mConfig.numBins - 1);
    long const rounded = std::lround(scaled);
    return static_cast<int32_t>(std::clamp<long>(rounded, 0, static_cast<long>(mConfig.numBins - 1)));
}

float DeltaTrajectoryTokenizer::extractYaw(Mat3 const& rotation)
{
    return std::atan2(rotation[0][1], rotation[0][0]);
}

float DeltaTrajectoryTokenizer::wrapAngleToPi(float angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

} // namespace traj
} // namespace trt_edgellm
