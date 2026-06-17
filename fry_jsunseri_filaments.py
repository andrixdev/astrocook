# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros


# James Sunseri
def textufy_james_sunseri_gas_xyzrho(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["dx", "log", "LQ"], ["rho", "log", "LQ"] ]
	box_center_x = 0
	box_center_y = 0
	box_center_z = 0
	box_radius = 0.008
	zoombox = [ box_center_x, box_center_y, box_center_z, box_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_radius, box_center_x + box_radius], [box_center_y - box_radius, box_center_y + box_radius], [box_center_z - box_radius, box_center_z + box_radius], [-7, -1], [-4, 7] ]
	kept_dimensions = [1, 1, 1, 0, 1]
	file_prefix = "xyzrho"
	
	source_file = "./data/jamessunseri/1-frame/MDG_gas.h5"
	file_type_token = "HDF5"
	dest_path = "jamessunseri/1-frame/"
	dest_file_name = "jamessunseri-gas-zoomed-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/13 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)
# textufy_james_sunseri_gas_xyzrho()

def textufy_james_sunseri_stars_xyzmass(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["mass", "log", "LQ"] ]
	box_center_x = 0
	box_center_y = 0
	box_center_z = 0
	box_radius = 0.008
	zoombox = [ box_center_x, box_center_y, box_center_z, box_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_radius, box_center_x + box_radius], [box_center_y - box_radius, box_center_y + box_radius], [box_center_z - box_radius, box_center_z + box_radius], [-14, -12] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./data/jamessunseri/1-frame/MDG_stars.h5"
	file_type_token = "HDF5"
	dest_path = "jamessunseri/1-frame/"
	dest_file_name = "jamessunseri-stars-zoomed-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/4 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)
# textufy_james_sunseri_stars_xyzmass()
