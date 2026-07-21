# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

# Fred Thompson star cluster
def textufy_fred_thompson_starcluster_gas_xyzrho(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["dx", "log", "LQ"], ["rho", "log", "LQ"] ]
	minmaxs = [ [-2000, 2000], [-2000, 2000], [-2000, 2000], [-5, 5], [-31, -20] ]
	kept_dimensions = [1, 1, 1, 0, 1]
	file_prefix = "xyzrho"
	
	source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_gas.h5"
	file_type_token = "HDF5"
	dest_path = "fredthompson/1-frame/"
	dest_file_name = "fredthompson-gas-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = False
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)

def textufy_fred_thompson_starcluster_stars_xyzmass(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["mass", "log", "LQ"] ]
	minmaxs = [ [-2000, 2000], [-2000, 2000], [-2000, 2000], [-3, 3.5] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_stars.h5"
	file_type_token = "HDF5"
	dest_path = "fredthompson/1-frame/"
	dest_file_name = "fredthompson-stars-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/2 if not is_test else 1/80 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = True
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)

def textufy_fred_thompson_starcluster_clusters_xyzmass(is_test=False):
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["id", "linear", "LQ"], ["mass", "log", "LQ"] ]
	minmaxs = [ [-2000, 2000], [-2000, 2000], [-2000, 2000], [0, 600], [3, 7] ]
	kept_dimensions = [1, 1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_clusters.h5"
	file_type_token = "HDF5"
	dest_path = "fredthompson/1-frame/"
	dest_file_name = "fredthompson-clusters-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/2 if not is_test else 1/80 # 1/1 is full rendering
	nb_logs = 15
	is_scanning = True
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)

# if __name__ == "__main__":
	# textufy_fred_thompson_starcluster_gas_xyzrho()
	# textufy_fred_thompson_starcluster_stars_xyzmass()
	# textufy_fred_thompson_starcluster_clusters_xyzmass()