# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros, update_minmaxs_of_minmaxs, print_minmaxs_of_minmaxs

# Dwarfgal from David Whitworth
def textufy_dwarfgal_xyz(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"] ]

	kept_dimensions = [1, 1, 1]
	file_prefix = "xyz"
	file_type_token = "NUMPY"
	dest_path = "dwarfgal/500-frames/"
	
	start_index = 2047
	end_index = 2546

	testing_density = 1/3 # 1/1 is full rendering
	nb_logs = 1
	is_scanning = False
	is_exporting = True

	zoombox_radius = 25
	zoombox = [ 500, 500, 500, zoombox_radius ]
	minmaxs = [ [500 - zoombox_radius, 500 + zoombox_radius], [500 - zoombox_radius, 500 + zoombox_radius], [500 - zoombox_radius, 500 + zoombox_radius] ]

	# Prepare animation global scan
	minmaxs_of_minmaxs = [[float("inf"), float("-inf")] for _ in range(len(dimensions))]

	for i in range(0, end_index - start_index + 1):
		frame = start_index + i
		frame_index = prepend_zeros(frame, 4)
		source_file = "./data/dwarfgal/500-frames/data_for_alex_" + str(frame_index) + ".npy"
		dest_file_name = "dwarfgal-" + file_prefix + "-" + prepend_zeros(str(i + 1), 3) + ("-testing" if is_test else "")
	
		frame_minmaxs = particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting, zoombox)

		# Update minmaxs of minmaxs with latest minmax
		# minmaxs_of_minmaxs = update_minmaxs_of_minmaxs(minmaxs_of_minmaxs, frame_minmaxs)

	# Print final minmax
	# print_minmaxs_of_minmaxs(minmaxs_of_minmaxs, dimensions)

if __name__ == "__main__":
	textufy_dwarfgal_xyz()
