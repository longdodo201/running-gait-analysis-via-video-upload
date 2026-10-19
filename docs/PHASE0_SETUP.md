# Phase 0: Windows setup and validation

## Target and dependency choices

Target Windows x64 with Python 3.11 and CPU inference. The direct dependency pins are in requirements.txt:

- MediaPipe 1.1.1 (Pose Landmarker API)
- OpenCV Contrib 4.12.0.88
- NumPy 2.2.6
- Streamlit 1.64.0

As of 2026-10-19, PyPI lists Windows x64 wheels and Python 3.11 classifiers for MediaPipe 1.1.1; OpenCV 4.12.0.88 provides a Windows x64 abi3 wheel and requires NumPy >=2,<2.3 on Python 3.11; NumPy 2.2.6 has a CPython 3.11 Windows x64 wheel; Streamlit 1.64.0 requires Python >=3.10 and accepts NumPy >=1.23,<3. These constraints intersect. MediaPipe's PyPI development-status classifier is Alpha, so keep it behind an adapter and recheck before upgrades.

Use opencv-contrib-python only. Do not install opencv-python alongside it: both distributions provide the cv2 module. The 100 MB and 60 second MP4 limits are proposed application input limits, not MediaPipe limits; they are not enforced by this Phase 0 setup.

The requirements file pins direct dependencies. Transitive dependencies are resolved by pip and are not fully locked yet; generate a Python 3.11 lock only after the target runtime has been validated.

## Windows PowerShell setup

Install the official Python install manager from python.org or the Microsoft Store. The Python.org manager supports installing Python 3.11 even though recent 3.11 security releases no longer publish traditional Windows installers. Then:

    py install 3.11
    py -3.11 --version
    py -3.11 -c "import struct; print(struct.calcsize('P') * 8)"
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip check

The version check should print Python 3.11.x; the architecture check should print 64. If PowerShell blocks activation, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass in that terminal, then activate again. This only changes the current process.

## Model asset: source, version, license, and integrity

Initial candidate: Pose Landmarker Full, float16, asset version 1. The official versioned model URL is:

    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

Create a local models directory and download the asset explicitly:

    New-Item -ItemType Directory -Force models | Out-Null
    Invoke-WebRequest "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -OutFile "models\pose_landmarker_full.task"

Verify the exact asset with PowerShell:

    $expected = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
    $actual = (Get-FileHash "models\pose_landmarker_full.task" -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Pose model checksum mismatch: $actual" }
    Write-Output "Pose model SHA-256 verified: $actual"

The expected SHA-256 was computed from the downloaded official asset during Phase 0. A MediaPipe maintainer has confirmed that this specific full float16 v1 asset is Apache-2.0 licensed. The MediaPipe Python package is also listed as Apache-2.0 on PyPI. The model is not committed; .gitignore excludes downloaded .task and .tflite files. If model redistribution is added later, include the applicable Apache-2.0 license and upstream notices. The project itself still has no selected license; this setup does not create one.

## Environment verification

Run these from the repository root after downloading the model:

    python -m pip check
    python -c "import cv2, mediapipe, numpy, streamlit; print(mediapipe.__version__, cv2.__version__, numpy.__version__, streamlit.__version__)"
    python scripts\verify_environment.py
    python scripts\verify_environment.py --image "C:\path\to\person.jpg"

The verification script checks the model hash, imports the direct dependencies, initializes Pose Landmarker with the CPU delegate, and optionally runs image inference. It never downloads files or makes network calls. For an inference smoke test, use a local image with a clearly visible person. This tests model execution only; it does not validate running gait events or metrics.

## Phase 0 acceptance criteria

| Check | Acceptance criterion | Status from this environment |
|---|---|---|
| Python environment | Create and activate a Windows x64 Python 3.11 virtual environment. | **Not completed here.** The available interpreter was Python 3.12.10; installing the Python manager was not completed because WinGet required Microsoft Store source terms. `pip download` successfully resolved all four pinned direct wheels for CPython 3.11 / Windows x64, but that is not a Python 3.11 runtime test. |
| Dependencies | Install requirements.txt and pass pip check. | **Passed on Windows 11 x64 / Python 3.12.10** with the exact pins in this repository. Python 3.11 install remains to be run. |
| Import and model initialization | Import dependencies, verify model checksum, initialize Pose Landmarker on CPU. | **Passed on Windows 11 x64 / Python 3.12.10** using the listed asset. |
| Local inference smoke test | Run detect on a local still image and record execution/result without claiming gait validity. | **Passed on Python 3.12.10** using the image linked by the upstream MediaPipe Python sample notebook: one pose returned with 33 landmarks. No running video was tested. |
| Model provenance/license | Record exact official source, version, license, and hash. | **Verified** for the Full float16 v1 asset; SHA-256 matched the downloaded file. |
| Reporting gaps | Clearly identify checks not performed. | Python 3.11 execution and any application-level MP4 size/duration enforcement remain unverified. No app or sample video was assumed to exist. |

The validation above is recorded honestly: dependency installation and inference succeeded on Python 3.12.10, not Python 3.11. Before treating Phase 0 as fully accepted, repeat the setup and verification commands in a Windows x64 Python 3.11 environment. MP4 decoding, the proposed 100 MB/60 second limits, and any gait-specific quality or accuracy remain untested.

## Troubleshooting

- If MediaPipe or cv2 imports fail, verify the activated interpreter with python --version and confirm it is the virtual environment.
- If pip check reports a conflict, reinstall from the pinned requirements file in a fresh virtual environment; do not add a second OpenCV distribution.
- If the checksum differs, delete the downloaded asset and fetch it again from the versioned official URL. Do not use an unverified model file.
- If inference returns zero poses, the model still ran; try a clear image with one full person visible. A still image smoke test does not establish running-video performance.
- TensorFlow Lite may print delegate or feedback-manager warnings during local CPU inference. Record them, and treat an exception or missing model asset as a failed check.

## References

- [MediaPipe 1.1.1 package metadata and Windows wheels](https://pypi.org/project/mediapipe/1.1.1/)
- [OpenCV Contrib 4.12.0.88 package metadata](https://pypi.org/project/opencv-contrib-python/4.12.0.88/)
- [NumPy 2.2.6 package metadata](https://pypi.org/project/numpy/2.2.6/)
- [Streamlit 1.64.0 package metadata](https://pypi.org/project/streamlit/1.64.0/)
- [Pose Landmarker Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker)
- [Upstream confirmation of the Full float16 v1 model license](https://github.com/google-ai-edge/mediapipe/issues/6306)
- [MediaPipe Python sample notebook used for the image smoke test](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb)
- [Python install manager for Windows](https://docs.python.org/3/using/windows.html) and [Python 3.11.17 release information](https://www.python.org/downloads/release/python-31117/)
