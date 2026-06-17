# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

# Isolated Galaxy for testing before RUM 2026 (from Tine by Maxime Trebitsch)
def textufy_isolagal_stars_xyz(is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["y", "linear", "HQ"],
		["z", "linear", "HQ"],
		["mass", "log", "HQ"]
	]
	
	source_file = "./data/isolagal/1-frame/isolagal_stars.h5"
	file_type_token = "HDF5"
	dest_path = "isolagal/1-frame/"
	dest_file_name = "isolagal-stars-xyz-test" + ("-testing" if is_test else "")
	minmaxs = [ [-40, 40], [-40, 40], [-40, 40], [-5.5, -4.5] ]
	kept_dimensions = [1, 1, 1, 0]
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)

def textufy_isolagal_gas_xyz(is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["y", "linear", "HQ"],
		["z", "linear", "HQ"],
		["dx", "linear", "LQ"],
		["rho", "log", "LQ"]
	]
	
	source_file = "./data/isolagal/1-frame/isolagal_gas.h5"
	file_type_token = "HDF5"
	dest_path = "isolagal/1-frame/"
	dest_file_name = "isolagal-gas-xyz" + ("-testing" if is_test else "")
	minmaxs = [ [-50, 50], [-50, 50], [-50, 50], [0, 1], [-8, 1] ]
	kept_dimensions = [1, 1, 1, 1, 1]
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)

# if __name__ == "__main__":
	# textufy_isolagal_stars_xyz()
	# textufy_isolagal_gas_xyz()