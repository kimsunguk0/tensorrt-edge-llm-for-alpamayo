#pragma once

#include "common/npyUtils.h"

#include <array>
#include <vector>

namespace trt_edgellm
{
namespace examples
{

struct TrajectoryDecodeResult
{
    std::vector<std::array<float, 3>> predXYZ;
    std::vector<std::array<std::array<float, 3>, 3>> predRot;
};

TrajectoryDecodeResult decodeActionToTrajectory(std::vector<std::array<float, 2>> const& action,
    common::NpyArrayFloat32 const& egoHistoryXYZ, common::NpyArrayFloat32 const& egoHistoryRot, float accelMean = 0.0F,
    float accelStd = 1.0F, float curvatureMean = 0.0F, float curvatureStd = 1.0F, float dt = 0.1F,
    float vLambda = 1.0e-6F, float vRidge = 1.0e-4F);

} // namespace examples
} // namespace trt_edgellm
