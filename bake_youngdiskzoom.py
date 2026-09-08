# ANDRIX ® 2025-2026 🤙
# 
# Bakes the klodus using cube_klodufy

from loguru import logger
from astrocutlery.cube_klodufy import klodufy
from astrocutlery.utensils import prepend_zeros, update_minmaxs_of_minmaxs, print_minmaxs_of_minmaxs

# YOUNGDISKZOOM animation (multiple frames)
def klodufy_youngdiskzoom_rho_animation (is_test=False):
    dimensions = [ ["rho", "log"] ]
    minmaxs = [ [-7, -3] ]
    file_prefix = "density"

    file_type_token = "NUMPY"
    #size = 256 # Cubes have different sizes...
    quality = "high"
    dest_path = "youngdiskzoom/300-frames/"
    testing_density = 1/1 if not is_test else 1/10 # 1/1 is full rendering
    nb_logs = 20

    start_index = 0
    end_index = 299

    for i in range(0, end_index - start_index + 1):
        frame = start_index + i
        frame_index = prepend_zeros(frame, 5)
        source_file = "./input/youngdiskzoom/300-frames/cube_density_" + str(frame_index) + ".npy"
        dest_file_name = ""
        
        is_scanning = True
        is_exporting = False
        size = 256

        scan = klodufy(source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, is_scanning, is_exporting)
        
        is_scanning = False
        is_exporting = True

        minmaxs = scan[0]
        size = scan[1]
        dest_file_name = "youngdiskzoom-rho-" + str(size) + "-" + prepend_zeros(str(i + 1), 3) + ("-testing" if is_test else "")
        print("Frame", frame, "minmaxs:", minmaxs)
        klodufy(source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, is_scanning, is_exporting)
        

if __name__ == "__main__":
    is_test = False
    klodufy_youngdiskzoom_rho_animation(is_test)
