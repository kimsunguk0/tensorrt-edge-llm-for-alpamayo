#!/usr/bin/env python3
"""
Alpamayo UDP Bridge — reference path slice 추종 모드

외부 publisher가 ~10 Hz로 차량 앞 ~20 m 분량의 reference path slice를
ego-frame JSON으로 보내준다.

  packet: {"pred_xyz": [[x, y, z], ...]}   # (N,3) list
  수신 좌표계: x=forward(m), y=right(m)  ← publisher 측 규약
  내부 좌표계: x=forward(m), y=left(m)   ← openpilot body frame
  parse 시점에 y 부호를 뒤집어 내부적으로는 openpilot 규약으로 통일한다.

매 패킷이 그 시점 차량 위치 기준으로 잘려 들어오므로 anchor/world 변환 없이
받은 path를 그대로 ego-frame path로 사용해 20 Hz control loop에서 pure pursuit
으로 curvature를 산출한다.

종방향은 TARGET_SPEED_MPS(=15 km/h) 유지 P 제어.
"""
import datetime
import csv
import json
import math
import os
import socket
import time
import numpy as np

import cereal.messaging as messaging
from cereal import log
from cereal.messaging import PubMaster, SubMaster
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import smooth_value
from openpilot.selfdrive.controls.lib.local_world import LocalWorld
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS

# ── 설정 ──────────────────────────────────────────────
UDP_PORT = 5005
LOCAL_PATH_VIZ_PORT = 5007   # 수신한 ref path slice를 ego-frame viz로 미러
VEHICLE_VIZ_PORT = 5006      # LocalWorld 현재 pose + 6초 trail (viz only)
WORLD_PATH_VIZ_PORT = 5008   # 수신 path 를 LocalWorld 기준 world frame 으로 변환해 송신 (viz only)
RECV_BUF_SIZE = 65535

T_IDXS = np.array(ModelConstants.T_IDXS, dtype=np.float64)
X_IDXS = np.array(ModelConstants.X_IDXS, dtype=np.float64)
IDX_N = ModelConstants.IDX_N   # 33

# ── 종방향 ────────────────────────────────────────────
TARGET_SPEED_MPS = 15.0 / 3.6        # 15 km/h ≈ 4.17 m/s
LON_KP = 0.3
ACCEL_MIN = -3.5
ACCEL_MAX = 2.0

MIN_LAT_CONTROL_SPEED = 0.3          # 이 속도 이하에서는 직전 curvature 유지

# ── pure pursuit 파라미터 ────────────────────────────
PP_LOOKAHEAD_M = 5.0                 # 고정 look-ahead
PP_CURV_LIMIT = 0.2

# ── 디버그 로그 ──────────────────────────────────────
DEBUG_LOG_DIR = "/tmp"
DIAG_LOG_DIR = os.environ.get("UDP_BRIDGE_DIAG_LOG_DIR", DEBUG_LOG_DIR)
DIAG_LD_VALUES_M = tuple(
    sorted(
        {
            PP_LOOKAHEAD_M,
            10.0,
            *(
                float(token)
                for token in os.environ.get("UDP_BRIDGE_DIAG_LDS_M", "").split(",")
                if token.strip()
            ),
        }
    )
)
DIAG_CONTROL_EVERY_N = max(1, int(os.environ.get("UDP_BRIDGE_DIAG_CONTROL_EVERY_N", "1")))


def format_xy_points(x, y, prec=2):
    """numpy array (x, y) → '[(x0,y0), (x1,y1), ...]' 사람이 읽는 문자열."""
    fmt = f"%.{prec}f"
    return "[" + ", ".join(f"({fmt % xi},{fmt % yi})" for xi, yi in zip(x, y)) + "]"


def _as_int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _packet_meta(d):
    header = d.get("packet_header") if isinstance(d.get("packet_header"), dict) else {}
    direct_header = d.get("header") if isinstance(d.get("header"), dict) else {}
    if direct_header:
        header = {**direct_header, **header}

    t0_utc_ns = _as_int_or_none(d.get("t0_utc_ns"))
    source_t0_us = _as_int_or_none(d.get("t0_us"))
    if source_t0_us is None and t0_utc_ns is not None:
        source_t0_us = t0_utc_ns // 1000
    if source_t0_us is None:
        source_t0_us = _as_int_or_none(header.get("source_t0_us"))

    return {
        "udp_mode": d.get("udp_mode"),
        "label": d.get("label"),
        "clip_id": d.get("clip_id"),
        "sample_id": _as_int_or_none(d.get("sample_id", header.get("sample_id"))),
        "plan_seq": _as_int_or_none(d.get("front_frame_id", header.get("plan_seq"))),
        "tx_seq": _as_int_or_none(header.get("tx_seq")),
        "source_t0_us": source_t0_us,
        "tx_time_us": _as_int_or_none(header.get("tx_time_us")),
        "payload_actual_offset_s": _as_float_or_none(d.get("actual_offset_s", d.get("target_offset_s"))),
        "inference_time_s": _as_float_or_none(d.get("inference_time_s")),
        "coord_note": d.get("coordinate_note"),
        "packet_header": header,
    }


# ── JSON 패킷 파싱 ───────────────────────────────────
def parse_path_packet(data: bytes):
    """{"pred_xyz": [[x,y,z], ...]} 형태 ego-frame slice 수신.
    실패 시 None. 수신은 x=forward, y=right 규약이지만 내부적으로는
    openpilot body frame(y=left)로 통일해서 반환한다.
    """
    try:
        d = json.loads(data.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        cloudlog.warning(f"udp_bridge: invalid JSON ({e})")
        return None
    if not isinstance(d, dict):
        cloudlog.warning(f"udp_bridge: invalid JSON root type {type(d).__name__}")
        return None

    try:
        pred_xyz = np.asarray(d['pred_xyz'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        cloudlog.warning(f"udp_bridge: malformed packet ({e})")
        return None

    if pred_xyz.ndim != 2 or pred_xyz.shape[1] < 2 or pred_xyz.shape[0] < 2:
        cloudlog.warning(f"udp_bridge: bad pred_xyz shape {pred_xyz.shape}")
        return None

    x = pred_xyz[:, 0]
    raw_y = pred_xyz[:, 1]
    y = -raw_y   # 수신 y=right(+) → 내부 y=left(+) (openpilot 규약)

    return {
        'x': x,
        'y': y,
        'raw_y': raw_y,
        'N': int(x.shape[0]),
        'meta': _packet_meta(d),
    }


# ── pure pursuit (lateral) ───────────────────────────
def pure_pursuit_curvature(path_ego, v_ego, lookahead_m=None):
    """ego-frame path(x=fwd, y=left)에서 pure pursuit 으로 desired curvature 산출.

    1) ego (0,0) 을 기준점으로
    2) 그 기준점에서 L_d 떨어진 path 위 goal 점을 잡아
    3) κ = 2·Δy / L_d²  (y left(+) → κ CCW(+), openpilot desiredCurvature 규약)
    L_d 까지 닿는 점이 없으면 path 끝점을 goal 로.
    """
    x = path_ego['x']
    y = path_ego['y']

    L_d = float(PP_LOOKAHEAD_M if lookahead_m is None else lookahead_m)

    d = np.hypot(x, y)

    fwd_mask = x > 0.0
    candidates = np.where(fwd_mask & (d >= L_d))[0]
    if candidates.size > 0:
        goal_idx = int(candidates[0])
    elif fwd_mask.any():
        goal_idx = int(np.where(fwd_mask)[0][-1])
    else:
        goal_idx = int(np.argmin(d))

    L_d_eff = max(float(d[goal_idx]), 1e-3)
    y_goal = float(y[goal_idx])

    kappa = 2.0 * y_goal / (L_d_eff * L_d_eff)
    kappa = float(np.clip(kappa, -PP_CURV_LIMIT, PP_CURV_LIMIT))
    return kappa, goal_idx, L_d_eff


def _safe_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return value


def _packet_age_s_from_us(now_wall_us, timestamp_us):
    timestamp_us = _as_int_or_none(timestamp_us)
    if timestamp_us is None:
        return None
    return (int(now_wall_us) - timestamp_us) / 1_000_000.0


def _path_shape_metrics(path_ego):
    y = np.asarray(path_ego["y"], dtype=np.float64)
    raw_y = np.asarray(path_ego.get("raw_y", y), dtype=np.float64)
    x = np.asarray(path_ego["x"], dtype=np.float64)
    dy = np.diff(y)
    d2y = np.diff(y, n=2)
    return {
        "first_x_m": float(x[0]) if x.size else None,
        "first_y_m": float(y[0]) if y.size else None,
        "first_raw_y_m": float(raw_y[0]) if raw_y.size else None,
        "last_x_m": float(x[-1]) if x.size else None,
        "last_y_m": float(y[-1]) if y.size else None,
        "mean_y_m": float(np.mean(y)) if y.size else None,
        "std_y_m": float(np.std(y)) if y.size else None,
        "max_abs_y_m": float(np.max(np.abs(y))) if y.size else None,
        "dy_step_rms_m": float(np.sqrt(np.mean(dy * dy))) if dy.size else None,
        "d2y_step_rms_m": float(np.sqrt(np.mean(d2y * d2y))) if d2y.size else None,
    }


def build_path_diag_row(path_ego, *, event, recv_count, frame_id, v_ego, kappa_smoothed,
                        prev_goal_y_by_ld, now_wall_us, now_mono_s):
    meta = dict(path_ego.get("meta") or {})
    row = {
        "event": event,
        "monotonic_s": now_mono_s,
        "wall_unix_s": now_wall_us / 1_000_000.0,
        "frame_id": int(frame_id),
        "recv_count": int(recv_count),
        "N": int(path_ego.get("N", len(path_ego.get("x", [])))),
        "v_ego_mps": _safe_float(v_ego),
        "kappa_smoothed": _safe_float(kappa_smoothed),
        "udp_mode": meta.get("udp_mode") or "",
        "label": meta.get("label") or "",
        "clip_id": meta.get("clip_id") or "",
        "sample_id": "" if meta.get("sample_id") is None else int(meta.get("sample_id")),
        "plan_seq": "" if meta.get("plan_seq") is None else int(meta.get("plan_seq")),
        "tx_seq": "" if meta.get("tx_seq") is None else int(meta.get("tx_seq")),
        "source_t0_us": "" if meta.get("source_t0_us") is None else int(meta.get("source_t0_us")),
        "tx_time_us": "" if meta.get("tx_time_us") is None else int(meta.get("tx_time_us")),
        "source_age_s": _safe_float(_packet_age_s_from_us(now_wall_us, meta.get("source_t0_us"))),
        "tx_age_s": _safe_float(_packet_age_s_from_us(now_wall_us, meta.get("tx_time_us"))),
        "payload_actual_offset_s": _safe_float(meta.get("payload_actual_offset_s")),
        "inference_time_s": _safe_float(meta.get("inference_time_s")),
        "coord_note": meta.get("coord_note") or "",
    }
    row.update({key: _safe_float(value) for key, value in _path_shape_metrics(path_ego).items()})

    for ld_m in DIAG_LD_VALUES_M:
        kappa_pp, goal_idx, L_d_eff = pure_pursuit_curvature(path_ego, v_ego, lookahead_m=ld_m)
        goal_x = float(path_ego["x"][goal_idx])
        goal_y = float(path_ego["y"][goal_idx])
        prev_goal_y = prev_goal_y_by_ld.get(float(ld_m))
        tag = f"ld{ld_m:g}"
        row[f"{tag}_goal_idx"] = int(goal_idx)
        row[f"{tag}_goal_x_m"] = goal_x
        row[f"{tag}_goal_y_m"] = goal_y
        row[f"{tag}_raw_goal_y_m"] = float(path_ego.get("raw_y", path_ego["y"])[goal_idx])
        row[f"{tag}_L_eff_m"] = float(L_d_eff)
        row[f"{tag}_kappa_raw"] = float(kappa_pp)
        row[f"{tag}_goal_y_delta_m"] = "" if prev_goal_y is None else goal_y - float(prev_goal_y)
    return row


def diag_fieldnames():
    fields = [
        "event",
        "monotonic_s",
        "wall_unix_s",
        "frame_id",
        "recv_count",
        "N",
        "v_ego_mps",
        "kappa_smoothed",
        "udp_mode",
        "label",
        "clip_id",
        "sample_id",
        "plan_seq",
        "tx_seq",
        "source_t0_us",
        "tx_time_us",
        "source_age_s",
        "tx_age_s",
        "payload_actual_offset_s",
        "inference_time_s",
        "coord_note",
        "first_x_m",
        "first_y_m",
        "first_raw_y_m",
        "last_x_m",
        "last_y_m",
        "mean_y_m",
        "std_y_m",
        "max_abs_y_m",
        "dy_step_rms_m",
        "d2y_step_rms_m",
    ]
    for ld_m in DIAG_LD_VALUES_M:
        tag = f"ld{ld_m:g}"
        fields.extend(
            [
                f"{tag}_goal_idx",
                f"{tag}_goal_x_m",
                f"{tag}_goal_y_m",
                f"{tag}_raw_goal_y_m",
                f"{tag}_L_eff_m",
                f"{tag}_kappa_raw",
                f"{tag}_goal_y_delta_m",
            ]
        )
    return fields


def open_diag_csv():
    os.makedirs(DIAG_LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DIAG_LOG_DIR, f"udp_bridge_path_diag_{ts}.csv")
    fh = open(path, "w", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=diag_fieldnames(), extrasaction="ignore")
    writer.writeheader()
    cloudlog.warning(f"udp_bridge path diagnostics CSV: {path}")
    return path, fh, writer


# ── 종방향 ───────────────────────────────────────────
def longitudinal_accel(v_ego):
    """TARGET_SPEED 유지 P 제어."""
    a_cmd = LON_KP * (TARGET_SPEED_MPS - max(v_ego, 0.0))
    return float(np.clip(a_cmd, ACCEL_MIN, ACCEL_MAX))


# ── viz용 리샘플 (path → T_IDXS 33점) ────────────────
def resample_for_viz(path_ego):
    """ego-frame path(x,y)를 T_IDXS 33점 시계열로 리샘플 (viz only).
    arc-length를 TARGET_SPEED 기준 시간으로 환산해 보간. path 너머는 마지막 값 hold.
    """
    x = path_ego['x']; y = path_ego['y']
    dx = np.diff(x); dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s_path = np.concatenate([[0.0], np.cumsum(ds)])
    s_target = T_IDXS * TARGET_SPEED_MPS

    x_viz = np.interp(s_target, s_path, x).astype(np.float32)
    y_viz = np.interp(s_target, s_path, y).astype(np.float32)
    z_viz = np.zeros(IDX_N, dtype=np.float32)
    dx_viz = np.gradient(x_viz)
    dy_viz = np.gradient(y_viz)
    yaw = np.arctan2(dy_viz, np.maximum(dx_viz, 1e-3)).astype(np.float32)
    v = np.full(IDX_N, TARGET_SPEED_MPS, dtype=np.float32)
    vx = (v * np.cos(yaw)).astype(np.float32)
    vy = (v * np.sin(yaw)).astype(np.float32)
    return {
        'x': x_viz, 'y': y_viz, 'z': z_viz,
        'yaw': yaw, 'v': v, 'vx': vx, 'vy': vy,
    }


def default_resampled():
    zeros = np.zeros(IDX_N, dtype=np.float32)
    return {
        'x': zeros.copy(), 'y': zeros.copy(), 'z': zeros.copy(),
        'yaw': zeros.copy(), 'v': zeros.copy(),
        'vx': zeros.copy(), 'vy': zeros.copy(),
    }


def idle_action():
    return log.ModelDataV2.Action(
        desiredCurvature=0.0,
        desiredAcceleration=0.0,
        shouldStop=True,
    )


# ── 메시지 발행 ──────────────────────────────────────
def fill_xyzt(builder, t, x, y, z, x_std=None, y_std=None, z_std=None):
    builder.t = list(t) if not isinstance(t, list) else t
    builder.x = x.tolist() if hasattr(x, 'tolist') else list(x)
    builder.y = y.tolist() if hasattr(y, 'tolist') else list(y)
    builder.z = z.tolist() if hasattr(z, 'tolist') else list(z)
    if x_std is not None:
        builder.xStd = x_std.tolist()
    if y_std is not None:
        builder.yStd = y_std.tolist()
    if z_std is not None:
        builder.zStd = z_std.tolist()


def publish_messages(pm, rs, action, frame_id, v_ego):
    """modelV2 + drivingModelData + longitudinalPlan + driverAssistance 발행."""
    now_ns = int(time.monotonic() * 1e9)
    t_list = ModelConstants.T_IDXS

    zeros_33 = np.zeros(IDX_N, dtype=np.float32)
    low_std = np.full(IDX_N, 0.1, dtype=np.float32)

    # ── modelV2 ──
    modelv2_send = messaging.new_message('modelV2')
    modelv2_send.valid = True
    mv2 = modelv2_send.modelV2

    mv2.frameId = frame_id
    mv2.frameIdExtra = frame_id
    mv2.frameAge = 0
    mv2.frameDropPerc = 0.0
    mv2.timestampEof = now_ns
    mv2.modelExecutionTime = 0.0

    fill_xyzt(mv2.position, t_list, rs['x'], rs['y'], rs['z'],
              x_std=low_std, y_std=low_std, z_std=low_std)
    fill_xyzt(mv2.velocity, t_list, rs['vx'], rs['vy'], zeros_33)
    fill_xyzt(mv2.acceleration, t_list, zeros_33, zeros_33, zeros_33)
    fill_xyzt(mv2.orientation, t_list, zeros_33, zeros_33, rs['yaw'])
    fill_xyzt(mv2.orientationRate, t_list, zeros_33, zeros_33, zeros_33)

    mv2.action = action

    mv2.init('laneLines', 4)
    default_lane_y = [1.8, 1.8, -1.8, -1.8]
    for i in range(4):
        ll = mv2.laneLines[i]
        lane_y = np.full(IDX_N, default_lane_y[i], dtype=np.float32)
        fill_xyzt(ll, [], X_IDXS.astype(np.float32), lane_y, zeros_33)
    mv2.laneLineStds = [0.0, 0.0, 0.0, 0.0]
    mv2.laneLineProbs = [0.0, 0.0, 0.0, 0.0]

    mv2.init('roadEdges', 2)
    default_edge_y = [3.0, -3.0]
    for i in range(2):
        re = mv2.roadEdges[i]
        edge_y = np.full(IDX_N, default_edge_y[i], dtype=np.float32)
        fill_xyzt(re, [], X_IDXS.astype(np.float32), edge_y, zeros_33)
    mv2.roadEdgeStds = [0.0, 0.0]

    mv2.init('leadsV3', 3)
    for i in range(3):
        lead = mv2.leadsV3[i]
        lead_n = len(ModelConstants.LEAD_T_IDXS)
        lead.t = ModelConstants.LEAD_T_IDXS
        lead.x = [200.0] * lead_n
        lead.y = [0.0] * lead_n
        lead.v = [0.0] * lead_n
        lead.a = [0.0] * lead_n
        lead.xStd = [100.0] * lead_n
        lead.yStd = [100.0] * lead_n
        lead.vStd = [100.0] * lead_n
        lead.aStd = [100.0] * lead_n
        lead.prob = 0.0
        lead.probTime = ModelConstants.LEAD_T_OFFSETS[i]

    meta = mv2.meta
    meta.desireState = [0.0] * ModelConstants.DESIRE_LEN
    meta.desirePrediction = [0.0] * (ModelConstants.DESIRE_PRED_LEN * ModelConstants.DESIRE_PRED_WIDTH)
    meta.engagedProb = 1.0

    meta.init('disengagePredictions')
    dp = meta.disengagePredictions
    dp.t = ModelConstants.META_T_IDXS
    n_meta = len(ModelConstants.META_T_IDXS)
    dp.brakeDisengageProbs = [0.0] * n_meta
    dp.gasDisengageProbs = [0.0] * n_meta
    dp.steerOverrideProbs = [0.0] * n_meta
    dp.brake3MetersPerSecondSquaredProbs = [0.0] * n_meta
    dp.brake4MetersPerSecondSquaredProbs = [0.0] * n_meta
    dp.brake5MetersPerSecondSquaredProbs = [0.0] * n_meta
    dp.gasPressProbs = [1.0] * n_meta
    dp.brakePressProbs = [0.0] * n_meta

    meta.laneChangeState = log.LaneChangeState.off
    meta.laneChangeDirection = log.LaneChangeDirection.none
    meta.hardBrakePredicted = False

    mv2.confidence = log.ModelDataV2.ConfidenceClass.green

    # ── drivingModelData ──
    dmd_send = messaging.new_message('drivingModelData')
    dmd_send.valid = True
    dmd = dmd_send.drivingModelData

    dmd.frameId = frame_id
    dmd.frameIdExtra = frame_id
    dmd.frameDropPerc = 0.0
    dmd.modelExecutionTime = 0.0
    dmd.action = action

    xyz = np.stack([rs['x'], rs['y'], rs['z']], axis=1)
    coeffs = np.polynomial.polynomial.polyfit(T_IDXS, xyz, deg=ModelConstants.POLY_PATH_DEGREE)
    dmd.path.xCoefficients = coeffs[:, 0].tolist()
    dmd.path.yCoefficients = coeffs[:, 1].tolist()
    dmd.path.zCoefficients = coeffs[:, 2].tolist()

    dmd.laneLineMeta.leftY = -1.8
    dmd.laneLineMeta.leftProb = 0.0
    dmd.laneLineMeta.rightY = 1.8
    dmd.laneLineMeta.rightProb = 0.0

    dmd.meta.laneChangeState = log.LaneChangeState.off
    dmd.meta.laneChangeDirection = log.LaneChangeDirection.none

    # ── longitudinalPlan ──
    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = True
    lp = plan_send.longitudinalPlan
    lp.aTarget = float(action.desiredAcceleration)
    lp.shouldStop = bool(action.shouldStop)
    lp.allowBrake = True
    lp.allowThrottle = True
    lp.hasLead = False
    lp.speeds = [float(v_ego)]

    # ── driverAssistance ──
    assist_send = messaging.new_message('driverAssistance')
    assist_send.valid = True

    pm.send('modelV2', modelv2_send)
    pm.send('drivingModelData', dmd_send)
    pm.send('longitudinalPlan', plan_send)
    pm.send('driverAssistance', assist_send)


# ── viz 송신 (수신 path 를 world frame 으로 변환) ─────
def ego_to_world(x_ego, y_ego, viz_anchor):
    """ego-frame (x=fwd, y=LEFT) → world frame (NED). viz_anchor=(x0,y0,yaw0)."""
    x0, y0, yaw0 = viz_anchor
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    wx = x0 + c0 * x_ego + s0 * y_ego
    wy = y0 + s0 * x_ego - c0 * y_ego
    return wx, wy


def build_world_path(path_ego, viz_anchor):
    """수신 ego-frame path → world frame 점 list."""
    px = path_ego['x']
    py = path_ego['y']
    wx, wy = ego_to_world(px, py, viz_anchor)
    return [{"x": float(wx[i]), "y": float(wy[i])} for i in range(len(px))]


def send_world_path_viz(viz_sock, path_ego, viz_anchor, seq,
                         goal_ego_xy, i_goal, L_d_eff, kappa_raw):
    """확장된 trajectory_world 송신: world path + ego path + goal point + kappa.

    points       : world frame (server 가 display_anchor 빼서 전달)
    ego_points   : 원본 ego frame 좌표 (debug panel 용)
    goal_world   : world frame goal (server 가 anchor 빼서 전달)
    goal_ego     : ego frame goal + 부가 정보
    kappa_raw    : pure pursuit raw 결과 (smooth 전)
    """
    world_pts = build_world_path(path_ego, viz_anchor)
    gx, gy = ego_to_world(np.float64(goal_ego_xy[0]), np.float64(goal_ego_xy[1]), viz_anchor)
    ego_pts = [
        {"x": float(path_ego['x'][i]), "y": float(path_ego['y'][i])}
        for i in range(len(path_ego['x']))
    ]
    msg = {
        "type": "trajectory_world",
        "seq": int(seq),
        "num_points": len(world_pts),
        "dt_s": 0.0,
        "points": world_pts,
        "ego_points": ego_pts,
        "goal_world": {"x": float(gx), "y": float(gy)},
        "goal_ego": {
            "x": float(goal_ego_xy[0]),
            "y": float(goal_ego_xy[1]),
            "i": int(i_goal),
            "L_d_eff": float(L_d_eff),
        },
        "kappa_raw": float(kappa_raw),
    }
    try:
        viz_sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", WORLD_PATH_VIZ_PORT))
    except OSError:
        pass


# ── viz 송신 (vehicle trail) ─────────────────────────
def send_vehicle_viz(viz_sock, world, lp, frame_id):
    """LocalWorld 현재 pose + 6초 history를 viz(5006) 로 송신 (viz only)."""
    cur = world.current()
    if cur is None:
        return
    _, x, y, yaw = cur
    speed = float(np.hypot(lp.velocityDevice.x, lp.velocityDevice.y))
    accel = float(lp.accelerationDevice.x)

    vehicle_msg = {
        "type": "vehicle",
        "x": float(x), "y": float(y),
        "heading": float(yaw),
        "speed": speed,
        "accel": accel,
        "curvature": 0.0,
        "should_stop": False,
        "frame": int(frame_id),
    }
    try:
        viz_sock.sendto(json.dumps(vehicle_msg).encode(), ("127.0.0.1", VEHICLE_VIZ_PORT))
    except OSError:
        pass

    hist = world.history()
    if len(hist) >= 2:
        pts = [
            {"x": float(hx), "y": float(hy), "yaw": float(hyaw), "vel": speed, "curvature": 0.0}
            for (_, hx, hy, hyaw) in hist
        ]
        traj_msg = {
            "type": "trajectory",
            "seq": int(frame_id),
            "plan_seq": int(frame_id),
            "coord_mode": 1,
            "num_points": len(pts),
            "dt_s": 0.05,
            "points": pts,
            "packet_count": int(frame_id),
        }
        try:
            viz_sock.sendto(json.dumps(traj_msg).encode(), ("127.0.0.1", VEHICLE_VIZ_PORT))
        except OSError:
            pass


# ── main ──────────────────────────────────────────────
def main():
    cloudlog.warning("udp_bridge init (ref-path slice mode, target=15km/h)")

    pm = PubMaster(["modelV2", "drivingModelData", "longitudinalPlan", "driverAssistance"])
    sm = SubMaster(["carState", "livePose", "selfdriveState"])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.setblocking(False)

    viz_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    viz_sock.setblocking(False)

    cloudlog.warning(f"udp_bridge listening on port {UDP_PORT} (ref path slice JSON @ ~10Hz)")
    cloudlog.warning(f"udp_bridge viz: vehicle/trail→{VEHICLE_VIZ_PORT}, raw mirror→{LOCAL_PATH_VIZ_PORT}, "
                     f"world path→{WORLD_PATH_VIZ_PORT}")
    cloudlog.warning(f"udp_bridge debug log: engage 시 {DEBUG_LOG_DIR}/udp_bridge_debug_*.log 생성")

    world = LocalWorld()        # viz only (vehicle trail)
    frame_id = 0
    path = None
    recv_count = 0
    log_counter = 0
    prev_curvature = 0.0
    diag_path, diag_fh, diag_writer = open_diag_csv()
    prev_goal_y_by_ld = {float(ld_m): None for ld_m in DIAG_LD_VALUES_M}

    # engage rising/falling edge 마다 debug log 파일을 새로 열고 닫음
    debug_log = None
    prev_engaged = False

    loop_period = 1.0 / ModelConstants.MODEL_RUN_FREQ  # 50ms = 20Hz

    while True:
        loop_start = time.monotonic()

        # 1. UDP 패킷 수신 — 큐에 쌓인 것 중 최신만 사용
        try:
            while True:
                data, _ = sock.recvfrom(RECV_BUF_SIZE)
                pkt = parse_path_packet(data)
                if pkt is not None:
                    recv_count += 1
                    path = pkt
                    # 디버그 로그: 수신 path (내부 좌표계, y=left). 한 줄=한 path. engage 중에만.
                    if debug_log is not None:
                        debug_log.write(
                            f"[t={time.monotonic():.3f}] PATH recv #{recv_count} N={pkt['N']} "
                            f"pts={format_xy_points(pkt['x'], pkt['y'])}\n"
                        )
                    # raw mirror → viz (ego-frame 원본)
                    try:
                        viz_sock.sendto(data, ('127.0.0.1', LOCAL_PATH_VIZ_PORT))
                    except OSError:
                        pass
                    # 패킷 도착 시점 pure pursuit 1회 → goal + raw kappa snapshot
                    v_ego_now = max(sm["carState"].vEgo, 0.0) if sm.alive["carState"] else 0.0
                    kappa_pp_pkt, i_goal_pkt, L_d_eff_pkt = pure_pursuit_curvature(pkt, v_ego_now)
                    now_wall_us = time.time_ns() // 1000
                    now_mono_s = time.monotonic()
                    diag_row = build_path_diag_row(
                        pkt,
                        event="recv",
                        recv_count=recv_count,
                        frame_id=frame_id,
                        v_ego=v_ego_now,
                        kappa_smoothed=None,
                        prev_goal_y_by_ld=prev_goal_y_by_ld,
                        now_wall_us=now_wall_us,
                        now_mono_s=now_mono_s,
                    )
                    diag_writer.writerow(diag_row)
                    for ld_m in DIAG_LD_VALUES_M:
                        tag = f"ld{ld_m:g}"
                        prev_goal_y_by_ld[float(ld_m)] = diag_row.get(f"{tag}_goal_y_m")
                    goal_xy = (float(pkt['x'][i_goal_pkt]), float(pkt['y'][i_goal_pkt]))
                    # world frame 변환 → viz(5008) (LocalWorld init 되어 있을 때만)
                    if world.is_initialized():
                        _, x0, y0, yaw0 = world.current()
                        send_world_path_viz(viz_sock, pkt, (x0, y0, yaw0), recv_count,
                                            goal_xy, i_goal_pkt, L_d_eff_pkt, kappa_pp_pkt)
        except BlockingIOError:
            pass

        # 2. SubMaster 업데이트 (LocalWorld 적분은 viz용 trail 에만 사용)
        sm.update(0)
        if sm.updated["livePose"]:
            world.update(sm["livePose"], sm.logMonoTime["livePose"])

        v_ego = max(sm["carState"].vEgo, 0.0)

        # 2.5. engage edge 감지 → debug log 파일 open/close
        engaged = bool(sm["selfdriveState"].enabled) if sm.alive["selfdriveState"] else False
        if engaged and not prev_engaged:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_log_path = f"{DEBUG_LOG_DIR}/udp_bridge_debug_{ts}.log"
            debug_log = open(debug_log_path, "w", buffering=1)   # line-buffered
            cloudlog.warning(f"udp_bridge debug log OPEN (engage): {debug_log_path}")
        elif (not engaged) and prev_engaged and debug_log is not None:
            cloudlog.warning(f"udp_bridge debug log CLOSE (disengage): {debug_log.name}")
            debug_log.close()
            debug_log = None
        prev_engaged = engaged

        # 3. tracker — 받은 ego-frame path 에 곧장 pure pursuit
        if path is not None:
            kappa_pp, i_goal, L_d_eff = pure_pursuit_curvature(path, v_ego)

            if v_ego > MIN_LAT_CONTROL_SPEED:
                kappa = smooth_value(kappa_pp, prev_curvature, LAT_SMOOTH_SECONDS)
            else:
                kappa = prev_curvature
            prev_curvature = kappa

            a_cmd = longitudinal_accel(v_ego)
            action = log.ModelDataV2.Action(
                desiredCurvature=float(kappa),
                desiredAcceleration=float(a_cmd),
                shouldStop=False,
            )
            rs = resample_for_viz(path)

            # 디버그 로그: 매 20Hz loop curvature, engage 중에만
            if debug_log is not None:
                debug_log.write(
                    f"[t={time.monotonic():.3f}] CURV frame={frame_id} v_ego={v_ego:.2f} "
                    f"raw={kappa_pp:+.4f} sm={kappa:+.4f} "
                    f"L_d_eff={L_d_eff:.2f} i_goal={i_goal} "
                    f"cte={float(path['y'][0]):+.2f} pkts={recv_count}\n"
                )

            if frame_id % DIAG_CONTROL_EVERY_N == 0:
                diag_writer.writerow(
                    build_path_diag_row(
                        path,
                        event="control",
                        recv_count=recv_count,
                        frame_id=frame_id,
                        v_ego=v_ego,
                        kappa_smoothed=kappa,
                        prev_goal_y_by_ld={},
                        now_wall_us=time.time_ns() // 1000,
                        now_mono_s=time.monotonic(),
                    )
                )

            log_counter += 1
            if log_counter % 20 == 1:   # 1Hz
                cte = float(path['y'][0])
                cloudlog.warning(
                    f"track: v_ego={v_ego:.2f} target={TARGET_SPEED_MPS:.2f} "
                    f"κ={kappa:+.4f}(raw {kappa_pp:+.4f}) a={a_cmd:+.2f} "
                    f"L_d={L_d_eff:.1f} i_goal={i_goal} N={path['N']} "
                    f"cte={cte:+.2f} pkts={recv_count}"
                )
        else:
            action = idle_action()
            rs = default_resampled()
            prev_curvature = 0.0

        # 4. 메시지 발행
        publish_messages(pm, rs, action, frame_id, v_ego)

        # 5. vehicle trail viz
        if world.is_initialized():
            send_vehicle_viz(viz_sock, world, sm["livePose"], frame_id)

        frame_id += 1

        # 6. 20Hz 타이밍 유지
        elapsed = time.monotonic() - loop_start
        sleep_time = loop_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cloudlog.warning("udp_bridge got SIGINT")
