# Alpamayo Raw Data Collection Task

This document is a handoff spec for building the raw data collection and offline sample-generation pipeline for Alpamayo-style training/inference data.

## Goal

Design and implement a raw data storage pipeline for a vehicle platform with:
- 4 cameras at 30 Hz
- IMU at 100 Hz
- GNSS/INS at 10 Hz

The immediate goal is **not** to create final Alpamayo `.npz` samples yet.
The first goal is to store raw sensor data in a way that is:
- compact
- replayable
- easy to synchronize offline
- compatible with later conversion into Alpamayo history/future samples

## Current Assumptions

- All sensors are already time-synchronized to **UTC**.
- This UTC synchronization should be preserved.
- We do **not** have proper calibration / extrinsics yet.
- We should still leave placeholders for calibration metadata.
- The platform currently has front-left, front, front-right, and front-tele cameras.
- A GNSS/INS fused navigation output may be available; if so, store it. If not, store GNSS raw fields.

## Important Design Decisions

### 1. Keep UTC sync
Store UTC timestamps for every sensor record.
Also store monotonic timestamps if available.

Recommended per-record time fields:
- `timestamp_utc_ns`
- `timestamp_monotonic_ns`

UTC is the master clock for cross-sensor alignment.
Monotonic time is for debugging, replay sanity, and driver-level ordering validation.

### 2. Do not downsample raw data too early
Raw storage should preserve original sensor rates:
- camera: 30 Hz
- IMU: 100 Hz
- GNSS/INS: 10 Hz

Do **not** reduce IMU to "nearest 3 samples per camera" in raw storage.
Do **not** expand GNSS to camera rate in raw storage.
Those are later derived features, not raw data.

### 3. Do not store camera frames as individual PNG/JPG files
At 30 Hz x 4 cameras, image-per-file storage becomes too heavy.
Instead:
- store each camera stream as chunked video
- store a frame index table separately

Recommended camera raw format:
- chunked `.mkv` or `.mp4`
- per-camera `frames.parquet` index file

### 4. Calibration is unavailable right now
We cannot depend on calibrated extrinsics today.
Still create placeholder metadata files so the dataset can be extended later without changing layout.

## Recommended Dataset Layout

```text
dataset_root/
  session_YYYYMMDD_run_0001/
    session_meta.json

    sensors/
      camera_front_left/
        chunks/
          chunk_0000.mkv
          chunk_0001.mkv
        frames.parquet

      camera_front/
        chunks/
          chunk_0000.mkv
          chunk_0001.mkv
        frames.parquet

      camera_front_right/
        chunks/
          chunk_0000.mkv
          chunk_0001.mkv
        frames.parquet

      camera_front_tele/
        chunks/
          chunk_0000.mkv
          chunk_0001.mkv
        frames.parquet

      imu/
        imu.parquet

      gnss_ins/
        gnss_ins.parquet

    calib/
      camera_intrinsics.json
      sensor_mounts_placeholder.json
```

## File-Level Requirements

### `session_meta.json`
Must contain:
- `session_id`
- `vehicle_id`
- `timezone`
- `time_sync.master_clock` = `utc`
- `time_sync.utc_source` if known (e.g. `gnss`)
- `time_sync.store_monotonic` = `true`
- `camera_names`
- `camera_fps`
- `imu_hz`
- `gnss_ins_hz`

Example:

```json
{
  "session_id": "session_20260327_run_0001",
  "vehicle_id": "test_car_01",
  "timezone": "UTC",
  "time_sync": {
    "master_clock": "utc",
    "utc_source": "gnss",
    "store_monotonic": true
  },
  "camera_names": ["front_left", "front", "front_right", "front_tele"],
  "camera_fps": 30,
  "imu_hz": 100,
  "gnss_ins_hz": 10
}
```

### Camera video chunks
Requirements:
- chunk each camera stream into manageable files (roughly 1 to 5 minutes each)
- preserve frame ordering
- codec choice should balance decode speed and storage efficiency
- prefer deterministic frame indexing over relying only on container timestamps

### Camera `frames.parquet`
Required columns:
- `frame_id`
- `chunk_id`
- `frame_index_in_chunk`
- `timestamp_utc_ns`
- `timestamp_monotonic_ns`
- `width`
- `height`

Optional but useful:
- `exposure_time_us`
- `gain`
- `dropped_frame`
- `trigger_id`

### `imu.parquet`
Required columns:
- `timestamp_utc_ns`
- `timestamp_monotonic_ns`
- `ax`
- `ay`
- `az`
- `gx`
- `gy`
- `gz`

Optional but useful:
- `temperature`
- `status`
- `seq`

### `gnss_ins.parquet`
Always store timestamps:
- `timestamp_utc_ns`
- `timestamp_monotonic_ns`

If only GNSS raw is available, store:
- `lat`
- `lon`
- `alt`
- `fix_type`
- `num_sats`
- `hdop`
- `vdop`
- covariance fields if available

If INS / fused navigation output is available, also store:
- `x_local`, `y_local`, `z_local` or equivalent local-frame position
- orientation: quaternion or roll/pitch/yaw
- `vx`, `vy`, `vz`
- `ins_status`
- covariance / quality flags if available

## Calibration Placeholder Files

### `camera_intrinsics.json`
Even if values are incomplete, define the schema now.
Store per camera:
- image size
- focal length / principal point if known
- distortion model if known

### `sensor_mounts_placeholder.json`
This is **not** a final calibration file.
It is only a placeholder for sensor identity and rough mounting notes.

Example:

```json
{
  "camera_front_left": {
    "description": "front-left windshield area",
    "nominal_fov_deg": 120
  },
  "camera_front": {
    "description": "front center",
    "nominal_fov_deg": 120
  },
  "camera_front_right": {
    "description": "front-right windshield area",
    "nominal_fov_deg": 120
  },
  "camera_front_tele": {
    "description": "front tele",
    "nominal_fov_deg": 30
  },
  "imu": {
    "description": "vehicle body mounted"
  },
  "gnss_ins": {
    "description": "roof antenna or fused navigation output"
  }
}
```

## What Not To Do

Do **not** do any of the following in raw storage:
- save all camera frames as individual image files
- reduce IMU raw data to only 3 samples nearest each camera frame
- replicate GNSS samples to camera rate
- generate Alpamayo `ego_history_*` or `ego_future_*` tensors directly in the raw tier
- assume calibrated extrinsics are available

## Expected Derived Pipeline Later

Raw storage should make the following later steps easy:
1. build a 10 Hz master timeline from UTC
2. select 4 camera frames around each `t0`
3. align IMU / GNSS / INS to each `t0`
4. generate fused ego pose history/future
5. export Alpamayo-ready samples:
   - `image_frames`
   - `camera_indices`
   - `ego_history_xyz`
   - `ego_history_rot`
   - `ego_future_xyz`
   - `ego_future_rot`
   - `relative_timestamps`
   - `absolute_timestamps`
   - `t0_us`
   - `clip_id`
   - `camera_order`

## Suggested Implementation Tasks

### Phase 1: schema + writer
Implement:
- session directory creator
- camera chunk writer
- camera frame index writer
- IMU parquet writer
- GNSS/INS parquet writer
- metadata writer

### Phase 2: validation
Implement checks for:
- monotonic timestamp ordering
- missing frames / dropped frames
- expected rate range per sensor
- chunk/frame index consistency
- camera chunk decode sanity

### Phase 3: offline sample index prototype
Do not generate full Alpamayo `.npz` yet.
Instead, prepare a `sample_index_10hz.parquet` prototype that maps:
- `t0_utc_ns`
- nearest camera frame IDs
- nearest GNSS/INS row IDs
- IMU row ranges

## Deliverables

The implementation should produce:
1. a directory layout exactly or very close to the schema above
2. a short README for the raw data format
3. example output for one short session
4. a validation script that prints dataset integrity statistics

## Success Criteria

We consider this task successful if:
- raw storage preserves all sensor rates
- UTC-based cross-sensor alignment is preserved
- camera storage is compact and indexed
- IMU/GNSS/INS are stored losslessly enough for later fusion / interpolation
- the dataset can be converted later into Alpamayo 10 Hz samples without redesigning the raw format
