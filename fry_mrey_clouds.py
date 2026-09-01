# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

# Maxime Rey molecular cloud
def textufy_maxime_rey_molecularcloud_gas_xyzrho(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["mass", "log", "LQ"] ]
	
	minmaxs = [ [0, 1], [0, 1], [0, 1], [-14, -12] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./input/maximereycloud/1-frame/stars.h5"
	file_type_token = "HDF5"
	dest_path = "maximereycloud/1-frame/"
	dest_file_name = "maximereycloud-gas-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = True
	is_exporting = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting) # textufy_maxime_rey_molecularcloud_gas_xyzrho()


# Maxime Rey new cloud
def textufy_maxime_rey_newcloud_xyzrho(is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["y", "linear", "HQ"],
		["z", "linear", "HQ"],
		["dx", "log", "LQ"],
		["rho", "log", "LQ"]
	]
	
	minmaxs = [ [0, 1e+21], [0, 1e+21], [0, 1e+21], [17.5, 19.5], [-28, -19] ]
	
	kept_dimensions = [1, 1, 1, 0, 1]
	file_prefix = "xyzrho"
	
	source_file = "./input/maximereynewcloud/1-frame/gas.h5"
	file_type_token = "HDF5"
	dest_path = "maximereynewcloud/1-frame/"
	dest_file_name = "maximereynewcloud-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = False
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)


# Maxime Rey new cloud animation (168 frames)
def textufy_maxime_rey_newcloud_xyzrho_frame(frame, index, is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["y", "linear", "HQ"],
		["z", "linear", "HQ"],
		["dx", "log", "LQ"],
		["rho", "log", "LQ"]
	]
	
	minmaxs = [ [0, 8e+20], [0, 8e+20], [0, 8e+20], [17.5, 19.5], [-29, -18] ]
	
	kept_dimensions = [1, 1, 1, 0, 1]
	file_prefix = "xyzrho"
	
	frame_index = prepend_zeros(frame, 5)
	source_file = "./input/maximereynewcloud/168-frames/output_" + frame_index + "/gas.h5"
	file_type_token = "HDF5"
	dest_path = "maximereynewcloud/168-frames/"
	dest_file_name = "maximereynewcloud-" + file_prefix + "-" + prepend_zeros(str(index), 3) + ("-testing" if is_test else "")
	testing_density = 1/100 # 1/1 is full rendering
	nb_logs = 2
	is_scanning = False
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)
def textufy_maxime_rey_newcloud_full_168_anim(is_test=False):
	logger.info("Generating 168 animation frames with positions and rho...")
	
	for i in range(0, 167 + 1):
		textufy_maxime_rey_newcloud_xyzrho_frame(i + 17, i + 1, is_test)
		
	logger.success("Generated 168 animation frames.")


if __name__ == "__main__":
	# textufy_maxime_rey_molecularcloud_gas_xyzrho()
	# textufy_maxime_rey_newcloud_xyzrho()
	# textufy_maxime_rey_newcloud_xyzrho_frame(17, 1)
	textufy_maxime_rey_newcloud_full_168_anim()