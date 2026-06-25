# Sourccey

Sourccey is integrated into DimOS as a native DIY robot package with three-camera ingest, IMU transport, base velocity control, sensor-debug tools, and optional spatial-memory workflows.

By default, the DimOS Sourccey connection treats these launchers as base-only workflows and untorques the arms during base control so DimOS does not silently reposition the arms while driving or debugging.

## What works today

- `sourccey-basic`
  - live robot connection
  - primary / companion / bottom camera streams
  - odom, IMU, and joint-state publishing
  - Rerun-compatible 2D/3D visualization when supported by the runtime
- `sourccey-sensor-debug`
  - sensor freshness / brightness / sharpness / odom / IMU health logging
- `sourccey-sensor-debug-view`
  - same debug stack plus Windows-native preview helper when launched through the provided Windows wrapper
- `sourccey-coordinator`
  - hooks Sourccey into the generic DimOS control coordinator
- `sourccey-coordinator-motion-test`
  - validates the full coordinator-to-base motion path
- `sourccey-survey-recorder`
  - records a multi-camera survey session for later mapping / reconstruction passes
- `sourccey-survey-scan`
  - runs the survey recorder plus an automated segmented base rotation for repeatable capture tests
  - default pattern centers the scan around the starting heading: `left:60,right:120,left:60`
- `sourccey-spatial`
  - enables DimOS spatial perception / memory when the optional perception dependencies are installed

## Recommended topology

- Robot host stays on Sourccey itself
- DimOS runs on a Windows workstation through WSL2
- Editing and Git stay on Windows
- Linux-first runtime dependencies stay inside WSL

## Robot host

Run the Sourccey host on the robot:

```bash
cd ~/Desktop/Projects/sourccey-desktop/modules/lerobot-vulcan

uv run -m lerobot.robots.sourccey.sourccey.sourccey.sourccey_host \
  --slam_eye_only_mode=true \
  --slam_three_camera_front_priority_mode=true \
  --bottom_camera_enabled=true \
  --bottom_camera_path="/dev/video8" \
  --slam_eye_camera_fps=20 \
  --slam_eye_loop_freq_hz=20 \
  --slam_eye_width=640 \
  --slam_eye_height=480 \
  --slam_eye_fourcc=MJPG \
  --slam_bottom_camera_fps=30 \
  --slam_bottom_width=320 \
  --slam_bottom_height=240 \
  --slam_bottom_fourcc=YUYV \
  --slam_eye_power_line_frequency=2 \
  --slam_eye_auto_exposure=1 \
  --slam_eye_exposure_dynamic_framerate=false \
  --slam_eye_exposure_time_absolute=700 \
  --slam_eye_gain=48 \
  --slam_input_enabled=true \
  --slam_input_endpoint="tcp://*:5560" \
  --slam_publish_eye_only_mode=false \
  --slam_publish_fps=20 \
  --slam_resize_width=640 \
  --slam_resize_height=480 \
  --slam_imu_enabled=true \
  --slam_imu_sample_rate_hz=52 \
  --slam_imu_max_samples_per_packet=6
```

## Windows launchers

From the Windows checkout:

```powershell
cd C:\Users\Theor\Documents\WebsiteCode\VulcanSlam\dimos
```

Basic connection:

```powershell
scripts\windows\Run-SourcceyBasic.cmd
```

Sensor debug with native Windows preview:

```powershell
scripts\windows\Run-SourcceySensorDebug.cmd
```

Coordinator wiring:

```powershell
scripts\windows\Run-SourcceyCoordinator.cmd
```

Coordinator motion test:

```powershell
scripts\windows\Run-SourcceyCoordinatorMotionTest.cmd
```

Survey recording:

```powershell
scripts\windows\Run-SourcceySurveyRecorder.cmd
```

Automated survey scan:

```powershell
scripts\windows\Run-SourcceySurveyScan.cmd
```

Spatial pipeline:

```powershell
scripts\windows\Run-SourcceySpatial.cmd
```

## Notes

- The Windows wrappers call into WSL automatically.
- `DIMOS_ROBOT_IP` can be set in the Windows shell to override the default robot IP.
- The native preview helper is intentionally separate from the WSL runtime because OpenCV GUI behavior is much more reliable on Windows than inside WSL.
- Base-driving Sourccey blueprints untorque the arms by default; if you ever want active DimOS arm control later, that should be introduced as a separate explicit workflow instead of piggybacking on the base connection.
