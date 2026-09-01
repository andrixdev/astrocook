# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

# Test file from Benoit Commerçon for bifluid simulation (early 2025)
def textufy_bifluid_xyzrho(is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["y", "linear", "HQ"],
		["z", "linear", "HQ"],
		["rho", "linear", "LQ"] # already in log in source data
	]
	
	source_file = "./input/bifluid/1-frame/bin1_bifluid_00045_clean3.txt"
	file_type_token = "TXT"
	dest_path = "bifluid/1-frame/"
	dest_file_name = "bifluid-xyzrho" + ("-testing" if is_test else "")
	minmaxs = [ [0, 4], [0, 4], [0, 4], [-23.5, -18] ]
	kept_dimensions = [1, 1, 1, 1]
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = False
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)

if __name__ == "__main__":
	textufy_bifluid_xyzrho()
