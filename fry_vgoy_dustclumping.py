# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

def textufy_valentin_goy_test_clumping(is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["rho", "log", "HQ"],
		["rhod", "log", "HQ"],
		["vx", "linear", "LQ"],
		["vy", "linear", "LQ"],
		["vz", "linear", "LQ"],
		["vdx", "linear", "LQ"],
		["vdy", "linear", "LQ"],
		["vdz", "linear", "LQ"],
		["Bx", "linear", "LQ"],
		["By", "linear", "LQ"],
		["Bz", "linear", "LQ"]
	]
	
	minmaxs = [ [0, 5e16], [-21, -16.5], [-23, -17.5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [0, 3.4e-4], [-1e-4, 1e-4], [-1e-4, 1e-4] ]
	
	kept_dimensions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	file_prefix = "all"
	
	source_file = "./data/valentingoy/1-frame-test/1D_magnetic_clumping_test.hdf5"
	file_type_token = "HDF5-GOY"
	dest_path = "valentingoy/1-frame-test/"
	dest_file_name = "valentingoy-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 20
	skip_scanning = False
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_valentin_goy_test_clumping()

def textufy_valentin_goy_hd_test_clumping(is_test=False):
	dimensions = [
		["x", "linear", "HQ"],
		["rho", "log", "LQ"],
		["rhod", "log", "LQ"],
		["vx", "linear", "HQ"],
		["vy", "linear", "HQ"],
		["vz", "linear", "HQ"],
		["vdx", "linear", "HQ"],
		["vdy", "linear", "HQ"],
		["vdz", "linear", "HQ"],
		["Bx", "linear", "HQ"],
		["By", "linear", "HQ"],
		["Bz", "linear", "HQ"],
		["sd", "log", "LQ"]
	]
	
	minmaxs = [ [0, 5e16], [-24, -15], [-24, -15], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [0, 3.4e-4], [-1e-4, 1e-4], [-1e-4, 1e-4], [-3, 0] ]
	
	kept_dimensions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	file_prefix = "allsd"
	
	source_file = "./data/valentingoy/1-frame-hd-test/1D_4096_test.hdf5"
	file_type_token = "HDF5-GOY"
	dest_path = "valentingoy/1-frame-hd-test/"
	dest_file_name = "valentingoy-" + file_prefix + ("-testing" if is_test else "")
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 20
	skip_scanning = False
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_valentin_goy_hd_test_clumping()

# def textufy_valentin_goy_103_anim_test():
# 	with h5py.File('1D_4096_103_frames_test.hdf5', 'r') as hdf:
# 		# Afficher la structure du fichier
# 		hdf.visit(print)

# 		# Lire des données spécifiques
# 		rhod = hdf['Alex/rhod'][:,:]#1D array
# 		sd = hdf['Alex/sd'][:,:]#1D array
# 		rho = hdf['Alex/rho'][:,:]#1D array
# 		vz = hdf['Alex/vz'][:,:]#1D array
# 		vy = hdf['Alex/vy'][:,:]#1D array
# 		vdz = hdf['Alex/vdz'][:,:]#1D array
# 		vdy = hdf['Alex/vdy'][:,:]#1D array
# 		Bz = hdf['Alex/Bz'][:,:]#1D array
# 		By = hdf['Alex/By'][:,:]#1D array
# 		vx = hdf['Alex/vx'][:,:]#1D array
# 		vdx = hdf['Alex/vdx'][:,:]#1D array
# 		group = hdf['Alex']

# 		sd_max = hdf['Alex/sd_max'][:]#1D array
# 		time = hdf['Alex/time'][:]#1D array


# 		rho_0 = group.attrs['rho_0']#Scalar
# 		rhod_0 = group.attrs['rhod_0']#Scalar

	