# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
import numpy as np
from astrocutlery import particles_textufy
from astrocutlery.utensils import configure_loguru, prepend_zeros
from astrocutlery.particles_textufy import compute_loop_variables, enrich_output_file_name, particles_scan, particles_export

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

def scan_time_array(time):
	print(time)

	log = ""
	for i in range(0, len(time) - 2):
		dif1 = time[i+1] - time[i]
		dif2 = time[i+2] - time[i+1]
		log += str(round(100 * (dif2 - dif1) / dif1)) + "%"
		log += ", "
	
	print(log)

def update_minmaxs_of_minmaxs(minmaxs_of_minmaxs, minmaxs):

	for d in range(0, len(minmaxs_of_minmaxs)):
		# Update min value
		if (minmaxs[d][0] < minmaxs_of_minmaxs[d][0]):
			minmaxs_of_minmaxs[d][0] = minmaxs[d][0]
			
		# Update max value
		if (minmaxs[d][1] > minmaxs_of_minmaxs[d][1]):
			minmaxs_of_minmaxs[d][1] = minmaxs[d][1]

	return minmaxs_of_minmaxs

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
		Bx = hdf['Alex/Bx'][:,:] #2D array
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
		# scan_time_array(time)

	# Start main loop to extract animation frames
	configure_loguru()
	testing_density = 1/1
	dest_path = "valentingoy/103-frames-draft/"
	file_prefix = "allsd"
	file_type_token = "HDF5"
	dimensions = [
		["x", "linear", "HQ"],
		["rho", "log", "LQ"],
		["rhod", "log", "LQ"],
		["vx", "linear", "LQ"],
		["vy", "linear", "LQ"],
		["vz", "linear", "LQ"],
		["vdx", "linear", "LQ"],
		["vdy", "linear", "LQ"],
		["vdz", "linear", "LQ"],
		["Bx", "linear", "LQ"],
		["By", "linear", "LQ"],
		["Bz", "linear", "LQ"],
		["sd", "log", "HQ"]
	]
	kept_dimensions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	minmaxs = [ [0, 5e16], [-24, -15], [-26.5, -15], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [0, 3.4e-4], [-1e-4, 1e-4], [-1e-4, 1e-4], [-3, 0] ]
	nb_logs = 6
	
	size = time.shape[0]
	for i in range(0, size):
		print(i)

		columns = [x[i], rho[i], rhod[i], vx[i], vy[i], vz[i], vdx[i], vdy[i], vdz[i], Bx[i], By[i], Bz[i], sd[i]]

		data = np.column_stack(columns)

		# Compact version of particles_textufy with this input data loop on 2D source data array
		dest_file_name = "valentingoy-" + file_prefix + "-" + prepend_zeros(str(i), 3)
		dest_file_name = enrich_output_file_name(dest_file_name, testing_density)
		loop_vars = compute_loop_variables(data, testing_density)
		step = loop_vars[0]
		actual_count = loop_vars[1]

		# Scan
		# particles_scan(data, actual_count, step, dimensions, file_type_token, nb_logs)

		# Export
		particles_export(data, actual_count, step, dimensions, kept_dimensions, minmaxs, file_type_token, nb_logs, dest_path, dest_file_name, zoombox=False)
	
# Version 1 -> first part of animation
# Version 2 -> second part of animation
# Version 98 -> export .txt file with time
# Version 99 -> export .txt file with sd_max
def textufy_valentin_goy_full_anim_part_version(which_part, which_version):
	import h5py

	source_file = ""
	start_index = 0
	end_index = 0
	if (which_part == 1 and which_version == 1):
		source_file = "./data/valentingoy/619-frames/Magnetic_clumping_4096_BeforeGrowth_110.hdf5"
		start_index = 1
		end_index = 110
	elif (which_part == 2 and which_version == 1):
		source_file = "./data/valentingoy/619-frames/Magnetic_clumping_4096_WithGrowth_509.hdf5"
		start_index = 111
		end_index = 619
	elif (which_part == 1 and which_version == 2):
		source_file = "./data/valentingoy/1987-frames/Magnetic_clumping_4096_BeforeGrowth_625.hdf5"
		start_index = 1
		end_index = 625
	elif (which_part == 2 and which_version == 2):
		source_file = "./data/valentingoy/1987-frames/Magnetic_clumping_4096_WithGrowth_1362.hdf5"
		start_index = 626
		end_index = 626 + 1362 - 1
	elif (which_part == 1 and (which_version == 98 or which_version == 99)): # It's actually still version 2 but we're just extracting time
		source_file = "./data/valentingoy/1987-frames/Magnetic_clumping_4096_BeforeGrowth_625.hdf5"
		end_index = 625
	elif (which_part == 2 and (which_version == 98 or which_version == 99)): # It's actually still version 2 but we're just extracting sd_max
		source_file = "./data/valentingoy/1987-frames/Magnetic_clumping_4096_WithGrowth_1362.hdf5"
		end_index = 1362

	with h5py.File(source_file, 'r') as hdf:
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
		Bx = hdf['Alex/Bx'][:,:] #2D array
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
		# scan_time_array(time)

	# Start main loop to extract animation frames
	configure_loguru()
	testing_density = 1/1
	dest_path = "valentingoy/" + ("619" if which_version == 1 else "1987") + "-frames/"
	file_prefix = "allsd"
	file_type_token = "HDF5"
	dimensions = [
		["x", "linear", "HQ"],
		["rho", "log", "LQ"],
		["rhod", "log", "LQ"],
		["vx", "linear", "LQ"],
		["vy", "linear", "LQ"],
		["vz", "linear", "LQ"],
		["vdx", "linear", "LQ"],
		["vdy", "linear", "LQ"],
		["vdz", "linear", "LQ"],
		["Bx", "linear", "LQ"],
		["By", "linear", "LQ"],
		["Bz", "linear", "LQ"],
		["sd", "log", "HQ"]
	]
	kept_dimensions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

	minmaxs = []
	if (which_version == 1):
		minmaxs = [ [0, 5e16], [-24, -15], [-26.5, -15], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [0, 3.4e-4], [-1e-4, 1e-4], [-1e-4, 1e-4], [-4, 0] ]
	elif (which_version == 2):
		minmaxs = [ [0, 4e16], [-24, -15], [-26.5, -15], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [-6e5, 6e5], [0, 2.6e-4], [-2.2e-4, 2.2e-4], [-2.2e-4, 2.2e-4], [-4, 0] ]
	else:
		logger.info("Not generating any minmaxs for this value of which_version: " + str(which_version))

	nb_logs = 3
	
	# print("TIME")
	# print(time)
	# size = time.shape[0]
	# print("SIZE")
	# print(size)

	# Print value in console to then paste in txt file
	if (which_version == 98): # It's actually still version 2 but we're just extracting time
		for i in range(0, end_index):
			print(str(time[i]))

		return
	
	elif (which_version == 99): # It's actually still version 2 but we're just extracting sd_max
		for i in range(0, end_index):
			print(str(sd_max[i]))

		return

	# Prepare animation global scan
	minmaxs_of_minmaxs = [[float("inf"), float("-inf")] for _ in range(len(dimensions))]

	for i in range(start_index, end_index + 1):
		data_index = i - start_index
		# print(data_index)
		columns = [x[data_index], rho[data_index], rhod[data_index], vx[data_index], vy[data_index], vz[data_index], vdx[data_index], vdy[data_index], vdz[data_index], Bx[data_index], By[data_index], Bz[data_index], sd[data_index]]

		data = np.column_stack(columns)

		# Compact version of particles_textufy with this input data loop on 2D source data array
		dest_file_name = "valentingoy-" + file_prefix + "-" + prepend_zeros(str(i), 3)
		dest_file_name = enrich_output_file_name(dest_file_name, testing_density)
		loop_vars = compute_loop_variables(data, testing_density)
		step = loop_vars[0]
		actual_count = loop_vars[1]

		# Scan
		# minmaxs = particles_scan(data, actual_count, step, dimensions, file_type_token, nb_logs)
		
		# Update minmaxs of minmaxs with latest minmax
		# minmaxs_of_minmaxs = update_minmaxs_of_minmaxs(minmaxs_of_minmaxs, minmaxs)

		# Export
		# particles_export(data, actual_count, step, dimensions, kept_dimensions, minmaxs, file_type_token, nb_logs, dest_path, dest_file_name, zoombox=False)

	# Print final minmax
	logger.info("Logging overall scanned minima and maxima...")
	for d in range(len(dimensions)):
		logger.bind(color="fg #DD5").trace("Overall Min value for " + str(dimensions[d][0]) + " is: " + str(minmaxs_of_minmaxs[d][0]))
		logger.bind(color="fg #DD5").trace("Overall Max value for " + str(dimensions[d][0]) + " is: " + str(minmaxs_of_minmaxs[d][1]))
	
def textufy_valentin_goy_619_anim():
	textufy_valentin_goy_full_anim_part_version(1, 1)
	# textufy_valentin_goy_full_anim_part_version(2, 1)

def textufy_valentin_goy_1987_anim():
	# textufy_valentin_goy_full_anim_part_version(1, 2)
	textufy_valentin_goy_full_anim_part_version(2, 2)

def print_valentin_goy_time_txt():
	# textufy_valentin_goy_full_anim_part_version(1, 98)
	textufy_valentin_goy_full_anim_part_version(2, 98)

def print_valentin_goy_sd_max_txt():
	# textufy_valentin_goy_full_anim_part_version(1, 99)
	textufy_valentin_goy_full_anim_part_version(2, 99)

if __name__ == "__main__":
	# textufy_valentin_goy_test_clumping()
	# textufy_valentin_goy_hd_test_clumping()
	# textufy_valentin_goy_103_anim_test()
	# textufy_valentin_goy_619_anim()
	# textufy_valentin_goy_1987_anim()
	# print_valentin_goy_time_txt()
	print_valentin_goy_sd_max_txt()
