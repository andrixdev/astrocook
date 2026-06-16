# ANDRIX ® 2025-2026 🤙
# 
# Bakes the klodus using cube_klodufy

from loguru import logger
from astrocutlery.cube_klodufy import klodufy
from astrocutlery.utensils import prepend_zeros

# EMMAAYCOBERRY BOX
def klodufy_emmaaycoberry_box ():

    dimensions = [ ["rho", "log"] ]
    minmaxs = [ [-11, 1] ]
    file_prefix = "density"

    source_file = "./data/emmaaycoberry/1-frame/pressure_map.dat"
    file_type_token = "DAT"
    size = 512
    quality = "high"
    dest_path = "emmaaycoberry/1-frame/"
    dest_file_name = "emmaaycoberry-box-rho-" + str(size)
    testing_density = 1/2 # 1/1 is full rendering
    nb_logs = 20
    skip_scanning = False
    
    klodufy(source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, skip_scanning)
klodufy_emmaaycoberry_box ()
