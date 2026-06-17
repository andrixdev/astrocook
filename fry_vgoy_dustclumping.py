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
	is_scanning = True
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)

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
	is_scanning = False
	is_exporting = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting)

def textufy_valentin_goy_103_anim_test():
	import h5py

	with h5py.File('./data/valentingoy/103-frames-draft/1D_4096_103_frames_draft.hdf5', 'r') as hdf:
		# Afficher la structure du fichier
		hdf.visit(print)

		# Lire des données spécifiques
		x = hdf['Alex/x'][:,:] #2D array
		rho = hdf['Alex/rho'][:,:] #2D array
		rhod = hdf['Alex/rhod'][:,:] #2D array
		vx = hdf['Alex/vx'][:,:] #2D array
		vz = hdf['Alex/vz'][:,:] #2D array
		vy = hdf['Alex/vy'][:,:] #2D array
		vdx = hdf['Alex/vdx'][:,:] #2D array
		vdz = hdf['Alex/vdz'][:,:] #2D array
		vdy = hdf['Alex/vdy'][:,:] #2D array
		Bx = hdf['Alex/By'][:,:] #2D array
		By = hdf['Alex/By'][:,:] #2D array
		Bz = hdf['Alex/Bz'][:,:] #2D array
		sd = hdf['Alex/sd'][:,:] #2D array
		group = hdf['Alex']

		sd_max = hdf['Alex/sd_max'][:] #1D array
		time = hdf['Alex/time'][:] #1D array

		rho_0 = group.attrs['rho_0'] #Scalar
		rhod_0 = group.attrs['rhod_0'] #Scalar

		print(sd.shape)
		print(sd[sd.shape[0] - 1])
		# data = sd[i]

		# Scan time intervals
		# print(time)
		# log = ""
		# for i in range(0, len(time) - 2):
		# 	dif1 = time[i+1] - time[i]
		# 	dif2 = time[i+2] - time[i+1]
		# 	log += str(round(100 * (dif2 - dif1) / dif1)) + "%"
		# 	log += ", "
		
		# print(log)




if __name__ == "__main__":
	# textufy_valentin_goy_test_clumping()
	textufy_valentin_goy_hd_test_clumping()
	# textufy_valentin_goy_103_anim_test()
