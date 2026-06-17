# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

# San Han galaxy cluster
def textufy_san_han_galaxy_cluster_xyzdensitytemp(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["density", "log", "LQ"], ["temperature", "log", "LQ"] ]
	
	box_center_x = 0.5168804
	box_center_y = 0.49409705
	box_center_z = 0.50810833
	box_radius = 0.003 / 2
	zoombox = [ box_center_x, box_center_y, box_center_z, box_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_radius, box_center_x + box_radius], [box_center_y - box_radius, box_center_y + box_radius], [box_center_z - box_radius, box_center_z + box_radius], [-7, 1], [-1, 10] ]
	
	kept_dimensions = [1, 1, 1, 1, 1]
	file_prefix = "xyzdensitytemp"
	
	source_file = "./data/sanhangalaxycluster/1-frame/nc_cluster.h5"
	file_type_token = "HDF5-SANHAN"
	dest_path = "sanhangalaxycluster/1-frame/"
	dest_file_name = "sanhangalaxycluster-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/3 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)

if __name__ == "__main__":
	textufy_san_han_galaxy_cluster_xyzdensitytemp()