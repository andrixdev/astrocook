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
def klodufy_maxime_lombart_collapse (is_test=False):

    source_file = "./data/maximelombart/1-frame-cube/data_cube_128_ramses.npy"
    file_type_token = "NUMPY-MLOMBART"
    size = 128
    quality = "high"
    dest_path = "maximelombart/1-frame-cube/"

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
    current_y = cubes["current_x"]
    current_z = cubes["current_x"]
    current = np.zeros((size, size, size))

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

    # klodu_scan(rho, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    minmaxs = [ [-19, -12.5] ]
    dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    # klodu_export(rho, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 2: v
    dimensions = [ ["v", "log"] ]
    file_prefix = "v"

    # klodu_scan(v, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    minmaxs = [ [4.2, 5.5] ]
    dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    # klodu_export(v, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 3: sd
    dimensions = [ ["sd", "log"] ]
    file_prefix = "sd"

    # klodu_scan(sd, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    minmaxs = [ [-4.7, -2.3] ]
    dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(sd, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 4: v_dust
    dimensions = [ ["vdust", "log"] ]
    file_prefix = "vdust"

    # klodu_scan(v_dust, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    minmaxs = [ [4.3, 5.5] ]
    dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(v_dust, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    # Cube 5: current
    dimensions = [ ["current", "log"] ]
    file_prefix = "current"

    # klodu_scan(current, log_ratio_text, base_count, actual_count, x_range, y_range, z_range, step, dimensions, nb_logs)

    minmaxs = [ [-1.5, 11] ]
    dest_file_name = "maxime-lombart-cube-" + file_prefix + "-" + str(size) + ("-testing" if is_test else "")

    klodu_export(current, log_ratio_text, actual_count, dest_path, dest_file_name, base_size, testing_density, size, minmaxs, quality, x_range, y_range, z_range, step, dimensions, nb_logs)

    
    # current


if __name__ == "__main__":
    klodufy_maxime_lombart_collapse()
