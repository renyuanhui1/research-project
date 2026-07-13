import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input COLMAP camera_trajectory.csv")
    parser.add_argument("--output", required=True, help="Output AirSim trajectory csv")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale COLMAP units to meters")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    required = ["image_name", "x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Move first pose to origin.
    cx = df["x"] - df["x"].iloc[0]
    cy = df["y"] - df["y"].iloc[0]
    cz = df["z"] - df["z"].iloc[0]

    # AirSim uses NED: x forward, y right, z down.
    # For your current COLMAP result, z is the main forward direction.
    airsim_x = cz * args.scale
    airsim_y = cx * args.scale
    airsim_z = -cy * args.scale

    out = pd.DataFrame({
        "frame": df["image_name"],
        "x": airsim_x,
        "y": airsim_y,
        "z": airsim_z,
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(f"Saved {len(out)} points to {output}")
    print("First:", out.iloc[0].to_dict())
    print("Last:", out.iloc[-1].to_dict())


if __name__ == "__main__":
    main()