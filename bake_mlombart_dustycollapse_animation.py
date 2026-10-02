# ANDRIX (R) 2026
#
# Bake the 128-cubed Maxime Lombart dusty-collapse animation for Unity.

import json
from pathlib import Path

import numpy as np
from loguru import logger

from astrocutlery.cube_klodufy import compute_loop_variables, klodu_export
from astrocutlery.utensils import prepend_zeros


MISSING_FRAME_INDICES = frozenset((41, 42, 43, 44))
FRAME_INDICES = [i for i in range(11, 166) if i not in MISSING_FRAME_INDICES]
FIELD_NAMES = ("rho", "v", "vdust", "sd", "B", "current")
INPUT_DIRECTORY = Path("input/maximelombart/151-frames")
OUTPUT_DIRECTORY = Path("output/maximelombart/151-frames")
QUALITY = "low"
SIZE = 128


def get_manifest_path(testing_density):
    testing_value = round(1 / testing_density)
    suffix = "" if testing_value == 1 else f"-1-in-{testing_value}"
    return OUTPUT_DIRECTORY / f"mlombart_151_frames_minmaxs_bounds{suffix}.txt"


def get_frame_path(frame_index):
    return INPUT_DIRECTORY / f"cube_128_output_{frame_index}.npy"


def load_fields(frame_index):
    """Load a snapshot and derive the six scalar physical fields to export."""
    source_file = get_frame_path(frame_index)
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing snapshot: {source_file}")

    cubes = np.load(source_file, allow_pickle=True).tolist()
    rho = cubes["gas_mass_density"]
    sd = cubes["s_peak"]
    v = np.linalg.vector_norm(
        np.stack((cubes["v_x_gas"], cubes["v_y_gas"], cubes["v_z_gas"])), axis=0
    )
    vdust = np.linalg.vector_norm(
        np.stack((cubes["v_x_s_peak"], cubes["v_y_s_peak"], cubes["v_z_s_peak"])), axis=0
    )
    current = np.linalg.vector_norm(
        np.stack((cubes["current_x"], cubes["current_y"], cubes["current_z"])), axis=0
    )
    B = np.linalg.vector_norm(
        np.stack(
            (
                0.5 * (cubes["B_left_x"] + cubes["B_right_x"]),
                0.5 * (cubes["B_left_y"] + cubes["B_right_y"]),
                0.5 * (cubes["B_left_z"] + cubes["B_right_z"]),
            )
        ),
        axis=0,
    )

    fields = {"rho": rho, "v": v, "vdust": vdust, "sd": sd, "B": B, "current": current}
    for field_name, field in fields.items():
        if field.shape != (SIZE, SIZE, SIZE):
            raise ValueError(f"{source_file}: {field_name} has shape {field.shape}, expected {(SIZE, SIZE, SIZE)}")
    return fields


def calculate_global_minmaxs(testing_density):
    """Calculate log-space bounds across every frame for each output field."""
    minmaxs = {field_name: [float("inf"), float("-inf")] for field_name in FIELD_NAMES}

    for frame_index in FRAME_INDICES:
        logger.info(f"Scanning frame {frame_index} for global bounds.")
        for field_name, field in load_fields(frame_index).items():
            _, _, _, _, x_range, _y_range, _z_range, step = compute_loop_variables(field, testing_density)
            sampled_indices = np.arange(x_range) * step
            sampled_field = field[np.ix_(sampled_indices, sampled_indices, sampled_indices)]
            positive_values = sampled_field[np.isfinite(sampled_field) & (sampled_field > 0)]
            if positive_values.size == 0:
                continue
            field_min = float(np.log10(positive_values.min()))
            field_max = float(np.log10(positive_values.max()))
            minmaxs[field_name][0] = min(minmaxs[field_name][0], field_min)
            minmaxs[field_name][1] = max(minmaxs[field_name][1], field_max)

    for field_name, bounds in minmaxs.items():
        if not np.isfinite(bounds).all():
            raise ValueError(f"No positive finite values found for {field_name} across the animation.")
    return minmaxs


def load_or_create_minmaxs(testing_density):
    manifest_path = get_manifest_path(testing_density)
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as manifest_file:
            minmaxs = json.load(manifest_file)
        if set(minmaxs) != set(FIELD_NAMES):
            raise ValueError(f"Invalid bounds manifest: {manifest_path}")
        logger.info(f"Using cached global bounds from {manifest_path}.")
        return minmaxs

    minmaxs = calculate_global_minmaxs(testing_density)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(minmaxs, manifest_file, indent=2)
        manifest_file.write("\n")
    logger.success(f"Saved global bounds to {manifest_path}.")
    return minmaxs


def klodufy_maxime_lombart_collapse_151_frames(is_test=False):
    """Export the full 151-frame, six-field dusty-collapse animation."""
    if not INPUT_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIRECTORY}")
    if not OUTPUT_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {OUTPUT_DIRECTORY}")
    for field_name in FIELD_NAMES:
        field_directory = OUTPUT_DIRECTORY / field_name
        if not field_directory.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {field_directory}")

    testing_density = 1 / 13 if is_test else 1
    minmaxs = load_or_create_minmaxs(testing_density)

    for frame_index in FRAME_INDICES:
        logger.info(f"Exporting frame {frame_index}.")
        fields = load_fields(frame_index)
        for field_name, field in fields.items():
            (
                log_ratio_text,
                base_size,
                _base_count,
                actual_count,
                x_range,
                y_range,
                z_range,
                step,
            ) = compute_loop_variables(field, testing_density)
            klodu_export(
                field,
                log_ratio_text,
                actual_count,
                f"maximelombart/151-frames/{field_name}/",
                f"maximelombart-cube-128-{prepend_zeros(frame_index, 3)}-{field_name}",
                base_size,
                testing_density,
                SIZE,
                [minmaxs[field_name]],
                QUALITY,
                x_range,
                y_range,
                z_range,
                step,
                [[field_name, "log"]],
                20,
            )


if __name__ == "__main__":
    klodufy_maxime_lombart_collapse_151_frames(is_test=False)