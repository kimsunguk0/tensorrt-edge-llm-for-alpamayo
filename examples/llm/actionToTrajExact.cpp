#include "actionToTrajExact.h"

#include "common/checkMacros.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace examples
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
using Vec3 = std::array<float, 3>;
using Mat3 = std::array<std::array<float, 3>, 3>;

struct HistoryData
{
    std::vector<Vec3> xyz;
    std::vector<Mat3> rot;
};

inline double wrapAngle(double x)
{
    return std::atan2(std::sin(x), std::cos(x));
}

inline double& mat(std::vector<double>& a, int n, int r, int c)
{
    return a[static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(c)];
}

inline double matc(std::vector<double> const& a, int n, int r, int c)
{
    return a[static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(c)];
}

void addThirdOrderDTD(std::vector<double>& lhs, int n, double scale)
{
    if (n < 4)
    {
        return;
    }
    int const coeffs[4] = {-1, 3, -3, 1};
    for (int row = 0; row < n - 3; ++row)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mat(lhs, n, row + i, row + j) += scale * static_cast<double>(coeffs[i] * coeffs[j]);
            }
        }
    }
}

bool choleskyInplace(std::vector<double>& a, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = matc(a, n, i, j);
            for (int k = 0; k < j; ++k)
            {
                sum -= matc(a, n, i, k) * matc(a, n, j, k);
            }
            if (i == j)
            {
                if (sum <= 0.0)
                {
                    return false;
                }
                mat(a, n, i, j) = std::sqrt(sum);
            }
            else
            {
                mat(a, n, i, j) = sum / matc(a, n, j, j);
            }
        }
        for (int j = i + 1; j < n; ++j)
        {
            mat(a, n, i, j) = 0.0;
        }
    }
    return true;
}

std::vector<double> choleskySolve(std::vector<double> const& lower, std::vector<double> const& rhs, int n)
{
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
    {
        double sum = rhs[static_cast<size_t>(i)];
        for (int j = 0; j < i; ++j)
        {
            sum -= matc(lower, n, i, j) * y[static_cast<size_t>(j)];
        }
        y[static_cast<size_t>(i)] = sum / matc(lower, n, i, i);
    }

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i)
    {
        double sum = y[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j)
        {
            sum -= matc(lower, n, j, i) * x[static_cast<size_t>(j)];
        }
        x[static_cast<size_t>(i)] = sum / matc(lower, n, i, i);
    }
    return x;
}

HistoryData parseHistory(common::NpyArrayFloat32 const& egoHistoryXYZ, common::NpyArrayFloat32 const& egoHistoryRot)
{
    check::check(!egoHistoryXYZ.shape.empty(), "egoHistoryXYZ shape must not be empty.");
    check::check(!egoHistoryRot.shape.empty(), "egoHistoryRot shape must not be empty.");
    check::check(egoHistoryXYZ.shape.back() == 3, "egoHistoryXYZ last dimension must be 3.");
    check::check(egoHistoryRot.shape.size() >= 2 && egoHistoryRot.shape[egoHistoryRot.shape.size() - 1] == 3
            && egoHistoryRot.shape[egoHistoryRot.shape.size() - 2] == 3,
        "egoHistoryRot trailing dimensions must be 3x3.");

    int64_t const histLen = egoHistoryXYZ.shape[egoHistoryXYZ.shape.size() - 2];
    int64_t const xyzBatch = static_cast<int64_t>(egoHistoryXYZ.data.size()) / (histLen * 3);
    int64_t const rotBatch = static_cast<int64_t>(egoHistoryRot.data.size()) / (histLen * 9);
    check::check(xyzBatch >= 1, "egoHistoryXYZ must contain at least one batch entry.");
    check::check(rotBatch >= 1, "egoHistoryRot must contain at least one batch entry.");

    HistoryData result;
    result.xyz.resize(static_cast<size_t>(histLen));
    result.rot.resize(static_cast<size_t>(histLen));

    size_t xyzBase = 0;
    size_t rotBase = 0;
    for (int64_t t = 0; t < histLen; ++t)
    {
        size_t xyzOffset = xyzBase + static_cast<size_t>(t * 3);
        result.xyz[static_cast<size_t>(t)]
            = {egoHistoryXYZ.data[xyzOffset], egoHistoryXYZ.data[xyzOffset + 1], egoHistoryXYZ.data[xyzOffset + 2]};

        size_t rotOffset = rotBase + static_cast<size_t>(t * 9);
        result.rot[static_cast<size_t>(t)] = {{{egoHistoryRot.data[rotOffset], egoHistoryRot.data[rotOffset + 1],
                                                   egoHistoryRot.data[rotOffset + 2]},
            {egoHistoryRot.data[rotOffset + 3], egoHistoryRot.data[rotOffset + 4], egoHistoryRot.data[rotOffset + 5]},
            {egoHistoryRot.data[rotOffset + 6], egoHistoryRot.data[rotOffset + 7], egoHistoryRot.data[rotOffset + 8]}}};
    }

    return result;
}

double estimateV0One(HistoryData const& history, double dt, double vLambda, double vRidge)
{
    int const hist = static_cast<int>(history.xyz.size());
    check::check(hist >= 2, "ego history must have at least 2 points.");
    int const n = hist - 1;

    std::vector<double> theta(static_cast<size_t>(hist), 0.0);
    double prevRaw = 0.0;
    for (int t = 0; t < hist; ++t)
    {
        double raw = std::atan2(static_cast<double>(history.rot[static_cast<size_t>(t)][1][0]),
            static_cast<double>(history.rot[static_cast<size_t>(t)][0][0]));
        if (t == 0)
        {
            theta[static_cast<size_t>(t)] = raw;
        }
        else
        {
            theta[static_cast<size_t>(t)]
                = theta[static_cast<size_t>(t - 1)] + wrapAngle(raw - prevRaw);
        }
        prevRaw = raw;
    }

    int const vdim = n + 1;
    std::vector<double> lhs(static_cast<size_t>(vdim) * static_cast<size_t>(vdim), 0.0);
    std::vector<double> rhs(static_cast<size_t>(vdim), 0.0);

    for (int t = 0; t < n; ++t)
    {
        double dx = static_cast<double>(history.xyz[static_cast<size_t>(t + 1)][0] - history.xyz[static_cast<size_t>(t)][0]);
        double dy = static_cast<double>(history.xyz[static_cast<size_t>(t + 1)][1] - history.xyz[static_cast<size_t>(t)][1]);
        double gx = (2.0 / dt) * dx;
        double gy = (2.0 / dt) * dy;

        double c0 = std::cos(theta[static_cast<size_t>(t)]);
        double c1 = std::cos(theta[static_cast<size_t>(t + 1)]);
        double s0 = std::sin(theta[static_cast<size_t>(t)]);
        double s1 = std::sin(theta[static_cast<size_t>(t + 1)]);

        mat(lhs, vdim, t, t) += c0 * c0 + s0 * s0;
        mat(lhs, vdim, t, t + 1) += c0 * c1 + s0 * s1;
        mat(lhs, vdim, t + 1, t) += c1 * c0 + s1 * s0;
        mat(lhs, vdim, t + 1, t + 1) += c1 * c1 + s1 * s1;

        rhs[static_cast<size_t>(t)] += c0 * gx + s0 * gy;
        rhs[static_cast<size_t>(t + 1)] += c1 * gx + s1 * gy;
    }

    double const smoothScale = vLambda / std::pow(dt, 6.0);
    addThirdOrderDTD(lhs, vdim, smoothScale);
    for (int i = 0; i < vdim; ++i)
    {
        mat(lhs, vdim, i, i) += vRidge;
    }

    check::check(choleskyInplace(lhs, vdim), "Failed Cholesky factorization in decodeActionToTrajectory.");
    auto velocity = choleskySolve(lhs, rhs, vdim);
    return velocity.back();
}

} // namespace

TrajectoryDecodeResult decodeActionToTrajectory(std::vector<std::array<float, 2>> const& action,
    common::NpyArrayFloat32 const& egoHistoryXYZ, common::NpyArrayFloat32 const& egoHistoryRot, float accelMean,
    float accelStd, float curvatureMean, float curvatureStd, float dt, float vLambda, float vRidge)
{
    check::check(!action.empty(), "Action trajectory must not be empty.");
    HistoryData const history = parseHistory(egoHistoryXYZ, egoHistoryRot);

    double const v0 = estimateV0One(history, dt, vLambda, vRidge);
    size_t const horizon = action.size();

    std::vector<double> velocity(horizon + 1, 0.0);
    std::vector<double> theta(horizon + 1, 0.0);
    velocity[0] = v0;
    theta[0] = 0.0;

    for (size_t t = 0; t < horizon; ++t)
    {
        double accel = static_cast<double>(action[t][0]) * accelStd + accelMean;
        double kappa = static_cast<double>(action[t][1]) * curvatureStd + curvatureMean;
        velocity[t + 1] = velocity[t] + accel * dt;
        theta[t + 1] = theta[t] + kappa * velocity[t] * dt + kappa * accel * (0.5 * dt * dt);
    }

    TrajectoryDecodeResult result;
    result.predXYZ.resize(horizon);
    result.predRot.resize(horizon);

    double xCum = 0.0;
    double yCum = 0.0;
    float const z0 = history.xyz.back()[2];
    double const halfDt = 0.5 * dt;
    for (size_t t = 0; t < horizon; ++t)
    {
        xCum += (velocity[t] * std::cos(theta[t]) + velocity[t + 1] * std::cos(theta[t + 1])) * halfDt;
        yCum += (velocity[t] * std::sin(theta[t]) + velocity[t + 1] * std::sin(theta[t + 1])) * halfDt;
        result.predXYZ[t] = {static_cast<float>(xCum), static_cast<float>(yCum), z0};

        float const c = static_cast<float>(std::cos(theta[t + 1]));
        float const s = static_cast<float>(std::sin(theta[t + 1]));
        result.predRot[t] = {{{c, -s, 0.0F}, {s, c, 0.0F}, {0.0F, 0.0F, 1.0F}}};
    }

    return result;
}

} // namespace examples
} // namespace trt_edgellm
