# ANDRIX ® 2026 🤙
# 
# Bakes the klodus using cube_klodufy

from loguru import logger
from astrocutlery.cube_klodufy import klodufy
from astrocutlery.utensils import prepend_zeros

# MAXIME LOMBART DUSTY COLLAPSE
def klodufy_maxime_lombart_collapse (is_test=False):
    dimensions = [ ["rho", "log"] ]
    minmaxs = [ [-11, 1] ]
    file_prefix = "density"

    source_file = "./data/maximelombart/1-frame-cube/data_cube_128_ramses.npy"
    file_type_token = "NUMPY-MLOMBART"
    size = 128
    quality = "high"
    dest_path = "maximelombart/1-frame-cube/"
    dest_file_name = "maxime-lombart-cube-rho-" + str(size) + ("-testing" if is_test else "")
    testing_density = 1/20 # 1/1 is full rendering
    nb_logs = 20
    is_scanning = True
    
    klodufy(source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, is_scanning)

if __name__ == "__main__":
    klodufy_maxime_lombart_collapse()
