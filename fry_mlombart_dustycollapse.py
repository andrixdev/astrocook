# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

def textufy_maxime_lombart_test_collapse(is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["y", "linear", "HQ"],
		["z", "linear", "HQ"],
		["rho", "log", "LQ"],
		["size", "log", "LQ"]
	]
	minmaxs = [ [-5000, 5000], [-5000, 5000], [-5000, 5000], [-20, -11], [-6, -1.5] ]
	kept_dimensions = [ 1, 1, 1, 1, 1 ]
	file_prefix = "rhosize"
	source_file = "./data/maximelombart/1-frame-test/collapse_data_ramses_test.npy"
	file_type_token = "NUMPY"
	dest_path = "maximelombart/1-frame-test/"
	dest_file_name = "maximelombart-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/1
	nb_logs = 15
	is_scanning = False
	is_exporting = True
	
	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)

def textufy_maxime_lombart_zoomed_test_collapse(is_test=False):
    dimensions = [
        ["x", "linear", "HQ"],
        ["y", "linear", "HQ"],
        ["z", "linear", "HQ"],
        ["rho", "log", "LQ"],
        ["size", "log", "LQ"]
    ]
    minmaxs = [ [-5000, 5000], [-5000, 5000], [-5000, 5000], [-20, -11], [-6, -1.5] ]

    box_center_x = 0
    box_center_y = 0
    box_center_z = 0
    box_virial_radius = 250
    zoombox = [ box_center_x, box_center_y, box_center_z, box_virial_radius ] # x_center, y_center, z_center, radius
    minmaxs = [ [box_center_x - box_virial_radius, box_center_x + box_virial_radius], [box_center_y - box_virial_radius, box_center_y + box_virial_radius], [box_center_z - box_virial_radius, box_center_z + box_virial_radius], [-20, -11], [-6, -1.5] ]

    kept_dimensions = [ 1, 1, 1, 1, 1 ]
    file_prefix = "rhosize"
    source_file = "./data/maximelombart/1-frame-test/collapse_data_ramses_test.npy"
    file_type_token = "NUMPY"
    dest_path = "maximelombart/1-frame-test/"
    dest_file_name = "maximelombart-" + file_prefix + "-zoomed-" + str(box_virial_radius) + ("-testing" if is_test else "")
    testing_density = 1/1
    nb_logs = 15
    is_scanning = True
    is_exporting = True
    
    particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting, zoombox)

if __name__ == "__main__":
    # textufy_maxime_lombart_test_collapse()
    textufy_maxime_lombart_zoomed_test_collapse()