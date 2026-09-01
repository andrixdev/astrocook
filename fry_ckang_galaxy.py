# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

# Cheonsu Kang big box
def textufy_cheonsukang_bigbox_xyzrho(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["rho", "log", "LQ"] ]
	box_center_x = 0.4987918675078839570
	box_center_y = 0.5031370643040111723
	box_center_z = 0.5003282700126294724
	box_virial_radius = 0.002485454994227498295
	zoombox = [ box_center_x, box_center_y, box_center_z, box_virial_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_virial_radius, box_center_x + box_virial_radius], [box_center_y - box_virial_radius, box_center_y + box_virial_radius], [box_center_z - box_virial_radius, box_center_z + box_virial_radius], [-3.5, 7] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzrho"
	
	source_file = "./input/cheonsukang/1-frame/cell_00373.sav"
	file_type_token = "SAV"
	dest_path = "cheonsukang/1-frame/"
	dest_file_name = "cheonsukang-bigbox-zoomed-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/40 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = True
	is_exporting = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting, zoombox)

def textufy_cheonsukang_bigbox_xyzvxvyvzrhopmetal(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["vx", "linear", "HQ"], ["vy", "linear", "HQ"], ["vz", "linear", "HQ"], ["rho", "log", "LQ"], ["p", "log", "LQ"], ["metal", "linear", "LQ"] ]
	box_center_x = 0.4987918675078839570
	box_center_y = 0.5031370643040111723
	box_center_z = 0.5003282700126294724
	box_virial_radius = 0.002485454994227498295
	zoombox = [ box_center_x, box_center_y, box_center_z, box_virial_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_virial_radius, box_center_x + box_virial_radius], [box_center_y - box_virial_radius, box_center_y + box_virial_radius], [box_center_z - box_virial_radius, box_center_z + box_virial_radius], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-3.5, 7], [-12, 0], [0, .1] ]
	kept_dimensions = [1, 1, 1, 1, 1, 1, 1, 1, 1]
	file_prefix = "xyzvxvyvzrhopmetal"

	source_file = "./input/cheonsukang/1-frame/cell_00373.sav"
	file_type_token = "SAV"
	dest_path = "cheonsukang/1-frame/"
	dest_file_name = "cheonsukang-bigbox-zoomed-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/4 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = False
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting, zoombox)

# if __name__ == "__main__":
	# textufy_cheonsukang_bigbox_xyzrho()
	# textufy_cheonsukang_bigbox_xyzvxvyvzrhopmetal()