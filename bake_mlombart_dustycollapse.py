# ANDRIX ® 2026 🤙
# 
# Bakes the klodus using cube_klodufy

import math
import numpy as np
from loguru import logger
from astrocutlery.cube_klodufy import prepare_data_cube, compute_loop_variables, klodu_scan, klodu_export
from astrocutlery.utensils import prepend_zeros

# MAXIME LOMBART DUSTY COLLAPSE
# Note: this data contains 20 cubes within the same dictionary
# Note: can't use klodufy but a special sequence of its internal functions
def klodufy_maxime_lombart_collapse (mode, is_test=False):

    do_scan = is_test

    if mode == "big":
        source_file = "./data/maximelombart/1-frame-big-cube/data_cube_256_ramses_output00145_simu_multifluid.npy"
        size = 256
        dest_path = "maximelombart/1-frame-big-cube/"
    else:
        source_file = "./data/maximelombart/1-frame-cube/data_cube_128_ramses_output00040_simu_multifluid.npy"
        size = 128
        dest_path = "maximelombart/1-frame-cube/"

    file_type_token = "NUMPY-MLOMBART"
    quality = "low"

    dimensions = [ ["dummy", "linear"] ]
    cubes = prepare_data_cube(source_file, file_type_token, dimensions)

    logger.info("Dictionary keys are: " + str(cubes.keys()))

    # Note: position cubes are only relevant in the axis they represent, all other dimensions are useless constant values. Anyway we don't need them.

    # Cube 1: gas density (1 -> 1)
    rho = cubes["gas_mass_density"]

    # Cube 2: gas velocity (3 -> 1)
    v_x = cubes["v_x_gas"]
    v_y = cubes["v_y_gas"]
    v_z = cubes["v_z_gas"]
    v = np.zeros((size, size, size))

    # Cube 3: size at peak of distribution (1 -> 1)
    sd = cubes["s_peak"]

    # Cube 4: velocity at peak (3 -> 1)
    vx_dust = cubes["v_x_s_peak"]
    vy_dust = cubes["v_y_s_peak"]
    vz_dust = cubes["v_z_s_peak"]
    v_dust = np.zeros((size, size, size))

    # Cube 5: electric current (3 -> 1)
    current_x = cubes["current_x"]
    current_y = cubes["current_y"]
    current_z = cubes["current_z"]
    current = np.zeros((size, size, size))

    # Cube 6: magnetic field magnitude (average of left and right components)
    B_x_left = cubes["B_left_x"]
    B_y_left = cubes["B_left_y"]
    B_z_left = cubes["B_left_z"]
    B_x_right = cubes["B_right_x"]
    B_y_right = cubes["B_right_y"]
    B_z_right = cubes["B_right_z"]
    B = np.zeros((size, size, size))

    logger.info("Computing 1-dimensional cubes from 3-dimensional datasets...")

    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                vx = v_x[i][j][k]
                vy = v_y[i][j][k]
                vz = v_z[i][j][k]

                v[i][j][k] = math.sqrt(vx*vx + vy*vy + vz*vz)

                vxd = vx_dust[i][j][k]
                vyd = vy_dust[i][j][k]
                vzd = vz_dust[i][j][k]

                v_dust[i][j][k] = math.sqrt(vxd*vxd + vyd*vyd + vzd*vzd)

                cx = current_x[i][j][k]
                cy = current_y[i][j][k]
                cz = current_z[i][j][k]

                current[i][j][k] = math.sqrt(cx*cx + cy*cy + cz*cz)

                bx = 0.5 * (B_x_left[i][j][k] + B_x_right[i][j][k])
                by = 0.5 * (B_y_left[i][j][k] + B_y_right[i][j][k])
                bz = 0.5 * (B_z_left[i][j][k] + B_z_right[i][j][k])
                B[i][j][k] = math.sqrt(bx*bx + by*by + bz*bz)

    logger.success("Computed 1-dimensional cubes from 3-dimensional datasets.")

    testing_density = 1/1 if not is_test else 1/9 # 1/1 is full rendering
    nb_logs = 20

    loop_vars = compute_loop_variables(rho, testing_density)
    log_ratio_text = loop_vars[0]
    base_size = loop_vars[1]
    base_count = loop_vars[2]
    actual_count = loop_vars[3]
    x_range = loop_vars[4]
    y_range = loop_vars[5]
    z_range = loop_vars[6]
    step = loop_vars[7]

    # Cube 1: rho
    dimensions = [ ["rho", "log"] ]
    file_prefix = "rho"

    if do_scan:
        klodu_scan(rho, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    if mode == "big":
        minmaxs = [ [-19.0, -12.1] ]
        dest_file_name = "maxime-lombart-big-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")
    else:
        minmaxs = [ [-19, -12.5] ]
        dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(rho, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 2: v
    dimensions = [ ["v", "log"] ]
    file_prefix = "v"

    if do_scan:
        klodu_scan(v, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    if mode == "big":
        minmaxs = [ [3.5, 5.7] ]
        dest_file_name = "maxime-lombart-big-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")
    else:
        minmaxs = [ [4.2, 5.5] ]
        dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(v, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 3: sd
    dimensions = [ ["sd", "log"] ]
    file_prefix = "sd"

    if do_scan:
        klodu_scan(sd, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    if mode == "big":
        minmaxs = [ [-4.6, -1.7] ]
        dest_file_name = "maxime-lombart-big-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")
    else:
        minmaxs = [ [-4.7, -2.3] ]
        dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(sd, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 4: v_dust
    dimensions = [ ["vdust", "log"] ]
    file_prefix = "vdust"

    if do_scan:
        klodu_scan(v_dust, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    if mode == "big":
        minmaxs = [ [3.4, 5.7] ]
        dest_file_name = "maxime-lombart-big-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")
    else:
        minmaxs = [ [4.3, 5.5] ]
        dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(v_dust, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 5: current
    dimensions = [ ["current", "log"] ]
    file_prefix = "current"

    if do_scan:
        klodu_scan(current, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    if mode == "big":
        minmaxs = [ [4.4, 11.3] ]
        dest_file_name = "maxime-lombart-big-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")
    else:
        minmaxs = [ [-1.5, 11] ]
        dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(current, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 6: magnetic field magnitude
    dimensions = [ ["B", "log"] ]
    file_prefix = "B"

    if do_scan:
        klodu_scan(B, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    if mode == "big":
        minmaxs = [ [-5.2, -1.1] ]
        dest_file_name = "maxime-lombart-big-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")
    else:
        minmaxs = [ [-5, -1] ]
        dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(B, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)


if __name__ == "__main__":
    is_test = False
    # klodufy_maxime_lombart_collapse("normal", is_test)
    klodufy_maxime_lombart_collapse("big", is_test)
