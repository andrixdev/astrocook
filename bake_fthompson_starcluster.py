# ANDRIX ® 2025-2026 🤙
# 
# Bakes the klodus using cube_klodufy

from loguru import logger
from astrocutlery.cube_klodufy import klodufy
from astrocutlery.utensils import prepend_zeros

# FRED THOMPSON STAR CLUSTER (has xyz so extracted in particles_textufy)
def klodufy_fredthompson_starcluster(is_test=False):

    dimensions = [ ["rho", "log"], ["x", "linear"], ["y", "linear"], ["z", "linear"] ]
    minmaxs = [ [-50, 50], [-50, 50], [-50, 50], [-50, 50] ]
    file_prefix = "density"

    source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_gas.h5"
    # source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_stars.h5"
    # source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_gas.h5"
    file_type_token = "HDF5"
    size = 256
    quality = "high"
    dest_path = "fredthompson/"
    dest_file_name = "fredthompson-starcluster-rho-" + str(size) + ("-testing" if is_test else "")
    testing_density = 1/10 # 1/1 is full rendering
    nb_logs = 20
    skip_scanning = False
    
    klodufy (source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, skip_scanning)

if __name__ == "__main__":
    klodufy_fredthompson_starcluster()
