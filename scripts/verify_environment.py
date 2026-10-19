from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

EXPECTED_MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local pose-estimation dependencies and model.")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    parser.add_argument("--image", type=Path, help="Optional local image containing a visible person for inference.")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    actual_sha256 = sha256_file(args.model)
    if actual_sha256 != EXPECTED_MODEL_SHA256:
        parser.error(f"Model SHA-256 mismatch: {actual_sha256}")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")

    import cv2
    import mediapipe as mp
    import numpy as np
    import streamlit

    print(f"mediapipe={mp.__version__}")
    print(f"opencv-contrib-python={version('opencv-contrib-python')}; cv2={cv2.__version__}")
    print(f"numpy={np.__version__}")
    print(f"streamlit={streamlit.__version__}")
    print(f"model_sha256={actual_sha256}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(args.model.resolve()),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("Pose Landmarker initialized with CPU delegate.")
        if args.image is not None:
            result = landmarker.detect(mp.Image.create_from_file(str(args.image.resolve())))
            counts = [len(pose) for pose in result.pose_landmarks]
            print(f"poses_detected={len(counts)}; landmarks_per_pose={counts}")
        else:
            print("Inference skipped; pass --image <path> to run a local inference smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
