# ANDRIX ® 2025-2026 🤙
# 
# Bakes the klodus using cube_klodufy

from loguru import logger
from astrocutlery.cube_klodufy import klodufy
from astrocutlery.utensils import prepend_zeros

# YOHANDUBOIS GALAXY
def klodufy_yohandubois_galaxy_rho (is_test=False):
    dimensions = [ ["rho", "log"] ]
    minmaxs = [ [-7, -3] ]
    file_prefix = "density"

    source_file = "./input/yohandubois/1-frame/cube_gasdensity_output_00070.dat"
    file_type_token = "DAT"
    size = 128
    quality = "high"
    dest_path = "yohandubois/1-frame/"
    dest_file_name = "yohandubois-galaxy-rho-" + str(size) + ("-testing" if is_test else "")
    testing_density = 1/1 if not is_test else 1/10 # 1/1 is full rendering
    nb_logs = 20
    is_scanning = True
    is_exporting = True
    
    klodufy (source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, is_scanning, is_exporting)

def klodufy_yohandubois_galaxy_bz(is_test=False):
    dimensions = [ ["bz", "linear"] ]
    minmaxs = [ [-0.00001, 0.00001] ]
    file_prefix = "bz"

    source_file = "./input/yohandubois/1-frame/cube_bz_output_00070.dat"
    file_type_token = "DAT"
    size = 256
    quality = "high"
    dest_path = "yohandubois/1-frame/"
    dest_file_name = "yohandubois-galaxy-bz-" + str(size) + ("-testing" if is_test else "")
    testing_density = 1/1 if not is_test else 1/10 # 1/1 is full rendering
    nb_logs = 20
    is_scanning = True
    is_exporting = True
    
    klodufy(source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, is_scanning, is_exporting)

# if __name__ == "__main__":
#     klodufy_yohandubois_galaxy_rho()
#     klodufy_yohandubois_galaxy_bz()