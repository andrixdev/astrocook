# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file calls particles_textufy() with various parameters to parse different particle data dumps and generate text files with the desired particle data for Unity

from loguru import logger
from astrocutlery import particles_textufy
from astrocutlery.utensils import prepend_zeros

def particles_textufy_disktilt():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["vx", "linear", "LQ"], ["vy", "linear", "LQ"], ["vz", "linear", "LQ"], ["rho", "log", "LQ"], ["soundspeed", "log", "LQ"] ]
	
	source_file = "./data/disktilt/disktilt_fulldump_0314.sham"
	file_type_token = "SHAMROCK"
	dest_path = "disktilt/test/"
	dest_file_name = "particles-disktilt-full-0314-xyzvxyzrhosound"
	minmaxs = [ [-1.8, 1.8], [-1.8, 1.8], [-1.8, 1.8], [-1E-3, 1E-3], [-1E-3, 1E-3], [-1E-3, 1E-3], [-10, -3.5], [-6.3, -5.3] ]
	testing_density = 1/100 # 1/1 is full rendering

	# source_file = "./data/disktilt/disktilt_dump_0098.sham"
	# file_type_token = "SHAMROCK"
	# dest_path = "disktilt/test/"
	# dest_file_name = "particles-disktilt-reduced-xyzvxyzrhosound"
	# minmaxs = [ [-1.8, 1.8], [-1.8, 1.8], [-1.8, 1.8], [-1E-3, 1E-3], [-1E-3, 1E-3], [-1E-3, 1E-3], [-6.5, -3.5], [-6.3, -5.3] ]
	# testing_density = 1/10 # 1/1 is full rendering
	
	nb_logs = 10
	skip_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, minmaxs, testing_density, nb_logs, skip_scanning)
# particles_textufy_disktilt()

# OBSOLETE
def particles_textufy_disktilt_frame(frame, index):
	logger.info("Generating frame " + str(frame) + " of index " + str(index))
	
	frame_index = prepend_zeros(frame, 4)
	source_file = "./data/disktilt/99-frames/dump_" + frame_index + ".sham"
	file_type_token = "SHAMROCK"
	dest_path = "disktilt/99-frames/"
	dest_file_name = "particles-disktilt-reduced-" + frame_index
	pos_only = True
	rho_logarithmic_mode = False
	soundspeed_logarithmic_mode = False
	min_pos = -2
	max_pos = 2
	min_vel = 0
	max_vel = 0
	min_rho = 0
	max_rho = 0
	min_soundspeed = 0
	max_soundspeed = 0
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 2
	skip_scanning = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, pos_only, rho_logarithmic_mode, soundspeed_logarithmic_mode, min_pos, max_pos, min_vel, max_vel, min_rho, max_rho, min_soundspeed, max_soundspeed, testing_density, nb_logs, skip_scanning)
def particles_textufy_disktilt_full_99_anim():
	logger.info("Generating 99 particles animation frames with positions...")
	
	for f in range(0, 98 + 1):
		particles_textufy_disktilt_frame(f, f)
		
	logger.success("Generated 99 animation frames.")
# particles_textufy_disktilt_full_99_anim()

def textufy_dwarfgal_frame(frame, index):
	logger.info("Generating frame " + str(frame) + " of index " + str(index))

	# dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["rho", "log", "LQ"], ["vol", "log", "LQ"], ["bx", "linear", "LQ"], ["by", "linear", "LQ"], ["bz", "linear", "LQ"], ["vx", "linear", "LQ"], ["vy", "linear", "LQ"], ["vz", "linear", "LQ"] ]
	# source_file = "./data/dwarfgal/1-frame/data_for_alex_xyzrhovolbxbybzvxvyvz.npy"
	# dest_file_name = "dwarfgal-xyzrhovolbxbybzvxvyvz"
	# minmaxs = [ [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-1, 10], [-9, 3], [-600, 600], [-600, 600], [-600, 600], [-500, 500], [-500, 500], [-500, 500] ]
	
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["rho", "log", "LQ"], ["vol", "log", "LQ"] ]
	
	source_file = "./data/dwarfgal/100-frames/data_for_alex_" + str(frame) + ".npy"
	file_type_token = "NUMPY"
	dest_path = "dwarfgal/100-frames/"
	output_index = prepend_zeros(index, 3)
	dest_file_name = "dwarfgal-xyzrhovol-" + str(output_index)
	minmaxs = [ [425, 575], [425, 575], [425, 575], [-1, 10], [-9, 5] ]
	kept_dimensions = [1, 1, 1, 1, 1]
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 3
	skip_scanning = False
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
def textufy_dwarfgal_full_100_anim():
	logger.info("Generating 100 animation frames with positions and rho...")
	
	i = 0
	for f in range(1250, 1349 + 1):
		i = i + 1
		textufy_dwarfgal_frame(f, i)
		
	logger.success("Generated 100 animation frames.")
# textufy_dwarfgal_full_100_anim()

def textufy_zoomin():
	# x y z (kpc) vx vy vz (km/s) rho (H) level mass (H + He) temp, level : level of refinement (12 (7min)->20), 8 volume scale between two levels
	
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["vx", "linear", "LQ"], ["vy", "linear", "LQ"], ["vz", "linear", "LQ"], ["rho", "log", "LQ"], ["level", "linear", "LQ"], ["mass", "linear", "LQ"], ["temp", "linear", "LQ"] ]
	kept_dimensions = [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 ]
	
	# source_file = "./data/zoomin/zoomin-dummy.txt"
	source_file = "./data/zoomin/rdr_00629_l20.hydro"
	file_type_token = "TXT"
	dest_path = "zoomin/1-frame/"
	dest_file_name = "zoomin-xyzvxvyvzrholvl"
	minmaxs = [ [2.4, 3.1], [-1.3, -0.5], [-0.4, 0.4], [-100, 250], [150, 500], [-200, 200], [-4, 9], [13, 21], [0, 4000], [0, 10000000] ]
	testing_density = 1/3 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False
	
	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_zoomin()

def textufy_binarydisk_frame(frame, index):
	# Multiple of 10 index
	if (frame % 10 == 0):
		dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["vx", "linear", "LQ"], ["vy", "linear", "LQ"], ["vz", "linear", "LQ"], ["h", "log", "LQ"], ["divv", "linear", "LQ"], ["dt", "linear", "LQ"] ]
		kept_dimensions = [ 1, 1, 1, 0, 0, 0, 1, 0, 0 ]
		minmaxs = [ [-200, 200], [-200, 200], [-200, 200], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-3, 4], [-10000, 10000], [0, 1000] ]
	else:
		# Other
		dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["h", "log", "LQ"] ]
		kept_dimensions = [ 1, 1, 1, 1 ]
		minmaxs = [ [-200, 200], [-200, 200], [-200, 200], [-3, 4] ]
		
	frame = prepend_zeros(str(frame), 5)
	index = prepend_zeros(str(index), 3)
	source_file = "./data/binarydisk/102-frames/orb0m02gprev_" + str(frame)
	file_type_token = "PHANTOM"
	dest_path = "binarydisk/102-frames/"
	dest_file_name = "particles-binarydisk-" + str(index)
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 2
	skip_scanning = True
	only_scanning = False
	
	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
def textufy_binarydisk_full_102_anim():
	start_index = 10#0
	end_index = 101
	diff = end_index - start_index

	logger.info("Generating " + str(diff) + " animation frames with density data...")
	
	i = start_index
	for f in range(start_index, end_index + 1):
		textufy_binarydisk_frame(f, i + 1)
		i = i + 1
		
	logger.success("Generated " + str(diff + 1) + " animation frames.")
# textufy_binarydisk_full_102_anim()

def textufy_fracturings_frame_xyz():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["hpart", "log", "HQ"], ["vx", "linear", "LQ"], ["vy", "linear", "LQ"], ["vz", "linear", "LQ"],  ["uint", "log", "HQ"] ]
	
	source_file = "./data/fracturings/1-frame/dump_0918.sham"
	file_type_token = "SHAMROCK"
	dest_path = "fracturings/1-frame/"
	dest_file_name = "fracturings-xyz-0918-test"
	minmaxs = [ [-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2], [-4, -1], [-0.001, 0.001], [-0.001, 0.001], [-0.001, 0.001], [-10, -6] ]
	kept_dimensions = [1, 1, 1, 0, 0, 0, 0, 0]
	testing_density = 1/20 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = False
	only_scanning = True
	# Scanning at 1/1 takes 40 minutes

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_fracturings_frame_xyz()
def textufy_fracturings_frame_xyzhvxvyvzu():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["hpart", "log", "HQ"], ["vx", "linear", "LQ"], ["vy", "linear", "LQ"], ["vz", "linear", "LQ"],  ["uint", "log", "HQ"] ]
	
	source_file = "./data/fracturings/1-frame/dump_0918.sham"
	file_type_token = "SHAMROCK"
	dest_path = "fracturings/1-frame/"
	dest_file_name = "fracturings-xyzhvxvyvzu-0918"
	minmaxs = [ [-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2], [-4, -1], [-0.001, 0.001], [-0.001, 0.001], [-0.001, 0.001], [-10, -6] ]
	kept_dimensions = [1, 1, 1, 1, 1, 1, 1, 1]
	testing_density = 1/100 # 1/1 is full rendering
	nb_logs = 10
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_fracturings_frame_xyzhvxvyvzu()

def textufy_isolagal_stars_xyz():
	dimensions = [
		["x", "linear", "HQ"],
		["y", "linear", "HQ"],
		["z", "linear", "HQ"],
		["mass", "log", "HQ"]
	]
	
	source_file = "./data/isolagal/1-frame/isolagal_stars.h5"
	file_type_token = "HDF5"
	dest_path = "isolagal/1-frame/"
	dest_file_name = "isolagal-stars-xyz-test"
	minmaxs = [ [-40, 40], [-40, 40], [-40, 40], [-5.5, -4.5] ]
	kept_dimensions = [1, 1, 1, 0]
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_isolagal_stars_xyz()
def textufy_isolagal_gas_xyz():
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
	dest_file_name = "isolagal-gas-xyz"
	minmaxs = [ [-50, 50], [-50, 50], [-50, 50], [0, 1], [-8, 1] ]
	kept_dimensions = [1, 1, 1, 1, 1]
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_isolagal_gas_xyz()

# Fred Thompson star cluster
def textufy_fred_thompson_starcluster_gas_xyzrho():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["dx", "log", "LQ"], ["rho", "log", "LQ"] ]
	minmaxs = [ [-2000, 2000], [-2000, 2000], [-2000, 2000], [-5, 5], [-31, -20] ]
	kept_dimensions = [1, 1, 1, 0, 1]
	file_prefix = "xyzrho"
	
	source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_gas.h5"
	file_type_token = "HDF5"
	dest_path = "fredthompson/1-frame/"
	dest_file_name = "fredthompson-gas-" + file_prefix
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_fred_thompson_starcluster_gas_xyzrho()
def textufy_fred_thompson_starcluster_stars_xyzmass():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["mass", "log", "LQ"] ]
	minmaxs = [ [-2000, 2000], [-2000, 2000], [-2000, 2000], [-3, 3.5] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_stars.h5"
	file_type_token = "HDF5"
	dest_path = "fredthompson/1-frame/"
	dest_file_name = "fredthompson-stars-" + file_prefix
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = False
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_fred_thompson_starcluster_stars_xyzmass()
def textufy_fred_thompson_starcluster_clusters_xyzmass():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["id", "linear", "LQ"], ["mass", "log", "LQ"] ]
	minmaxs = [ [-2000, 2000], [-2000, 2000], [-2000, 2000], [0, 600], [3, 7] ]
	kept_dimensions = [1, 1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./data/fredthompson/1-frame/H10cluster_8pc_output176_clusters.h5"
	file_type_token = "HDF5"
	dest_path = "fredthompson/1-frame/"
	dest_file_name = "fredthompson-clusters-" + file_prefix
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = False
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_fred_thompson_starcluster_clusters_xyzmass()

# Cheonsu Kang big box
def textufy_cheonsukang_bigbox_xyzrho():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["rho", "log", "LQ"] ]
	box_center_x = 0.4987918675078839570
	box_center_y = 0.5031370643040111723
	box_center_z = 0.5003282700126294724
	box_virial_radius = 0.002485454994227498295
	zoombox = [ box_center_x, box_center_y, box_center_z, box_virial_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_virial_radius, box_center_x + box_virial_radius], [box_center_y - box_virial_radius, box_center_y + box_virial_radius], [box_center_z - box_virial_radius, box_center_z + box_virial_radius], [-3.5, 7] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzrho"
	
	source_file = "./data/cheonsukang/1-frame/cell_00373.sav"
	file_type_token = "SAV"
	dest_path = "cheonsukang/1-frame/"
	dest_file_name = "cheonsukang-bigbox-zoomed-" + file_prefix
	testing_density = 1/40 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = False
	only_scanning = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)
# textufy_cheonsukang_bigbox_xyzrho()
def textufy_cheonsukang_bigbox_xyzvxvyvzrhopmetal():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["vx", "linear", "HQ"], ["vy", "linear", "HQ"], ["vz", "linear", "HQ"], ["rho", "log", "LQ"], ["p", "log", "LQ"], ["metal", "linear", "LQ"] ]
	box_center_x = 0.4987918675078839570
	box_center_y = 0.5031370643040111723
	box_center_z = 0.5003282700126294724
	box_virial_radius = 0.002485454994227498295
	zoombox = [ box_center_x, box_center_y, box_center_z, box_virial_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_virial_radius, box_center_x + box_virial_radius], [box_center_y - box_virial_radius, box_center_y + box_virial_radius], [box_center_z - box_virial_radius, box_center_z + box_virial_radius], [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [-3.5, 7], [-12, 0], [0, .1] ]
	kept_dimensions = [1, 1, 1, 1, 1, 1, 1, 1, 1]
	file_prefix = "xyzvxvyvzrhopmetal"

	source_file = "./data/cheonsukang/1-frame/cell_00373.sav"
	file_type_token = "SAV"
	dest_path = "cheonsukang/1-frame/"
	dest_file_name = "cheonsukang-bigbox-zoomed-" + file_prefix
	testing_density = 1/4 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)
# textufy_cheonsukang_bigbox_xyzvxvyvzrhopmetal()

# James Sunseri
def textufy_james_sunseri_gas_xyzrho():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["dx", "log", "LQ"], ["rho", "log", "LQ"] ]
	box_center_x = 0
	box_center_y = 0
	box_center_z = 0
	box_radius = 0.008
	zoombox = [ box_center_x, box_center_y, box_center_z, box_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_radius, box_center_x + box_radius], [box_center_y - box_radius, box_center_y + box_radius], [box_center_z - box_radius, box_center_z + box_radius], [-7, -1], [-4, 7] ]
	kept_dimensions = [1, 1, 1, 0, 1]
	file_prefix = "xyzrho"
	
	source_file = "./data/jamessunseri/1-frame/MDG_gas.h5"
	file_type_token = "HDF5"
	dest_path = "jamessunseri/1-frame/"
	dest_file_name = "jamessunseri-gas-zoomed-" + file_prefix
	testing_density = 1/13 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)
# textufy_james_sunseri_gas_xyzrho()
def textufy_james_sunseri_stars_xyzmass():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["mass", "log", "LQ"] ]
	box_center_x = 0
	box_center_y = 0
	box_center_z = 0
	box_radius = 0.008
	zoombox = [ box_center_x, box_center_y, box_center_z, box_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_radius, box_center_x + box_radius], [box_center_y - box_radius, box_center_y + box_radius], [box_center_z - box_radius, box_center_z + box_radius], [-14, -12] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./data/jamessunseri/1-frame/MDG_stars.h5"
	file_type_token = "HDF5"
	dest_path = "jamessunseri/1-frame/"
	dest_file_name = "jamessunseri-stars-zoomed-" + file_prefix
	testing_density = 1/4 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)
# textufy_james_sunseri_stars_xyzmass()

# Maxime Rey molecular cloud
def textufy_maxime_rey_molecularcloud_gas_xyzrho():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["mass", "log", "LQ"] ]
	
	minmaxs = [ [0, 1], [0, 1], [0, 1], [-14, -12] ]
	kept_dimensions = [1, 1, 1, 1]
	file_prefix = "xyzmass"
	
	source_file = "./data/maximereycloud/1-frame/stars.h5"
	file_type_token = "HDF5"
	dest_path = "maximereycloud/1-frame/"
	dest_file_name = "maximereycloud-gas-" + file_prefix
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = False
	only_scanning = True

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning) # textufy_maxime_rey_molecularcloud_gas_xyzrho()

# San Han galaxy cluster
def textufy_san_han_galaxy_cluster_xyzdensitytemp():
	dimensions = [ ["x", "linear", "HQ"], ["y", "linear", "HQ"], ["z", "linear", "HQ"], ["density", "log", "LQ"], ["temperature", "log", "LQ"] ]
	
	box_center_x = 0.5168804
	box_center_y = 0.49409705
	box_center_z = 0.50810833
	box_radius = 0.003 / 2
	zoombox = [ box_center_x, box_center_y, box_center_z, box_radius ] # x_center, y_center, z_center, radius
	minmaxs = [ [box_center_x - box_radius, box_center_x + box_radius], [box_center_y - box_radius, box_center_y + box_radius], [box_center_z - box_radius, box_center_z + box_radius], [-7, 1], [-1, 10] ]
	
	kept_dimensions = [1, 1, 1, 1, 1]
	file_prefix = "xyzdensitytemp"
	
	source_file = "./data/sanhangalaxycluster/1-frame/nc_cluster.h5"
	file_type_token = "HDF5-SANHAN"
	dest_path = "sanhangalaxycluster/1-frame/"
	dest_file_name = "sanhangalaxycluster-" + file_prefix
	testing_density = 1/3 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox)
# textufy_san_han_galaxy_cluster_xyzdensitytemp()

# Maxime Rey new cloud
def textufy_maxime_rey_newcloud_xyzrho():
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
	
	source_file = "./data/maximereynewcloud/1-frame/gas.h5"
	file_type_token = "HDF5"
	dest_path = "maximereynewcloud/1-frame/"
	dest_file_name = "maximereynewcloud-" + file_prefix
	testing_density = 1/1 # 1/1 is full rendering
	nb_logs = 15
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_maxime_rey_newcloud_xyzrho()

# Maxime Rey new cloud animation (168 frames)
def textufy_maxime_rey_newcloud_xyzrho_frame(frame, index):
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
	source_file = "./data/maximereynewcloud/168-frames/output_" + frame_index + "/gas.h5"
	file_type_token = "HDF5"
	dest_path = "maximereynewcloud/168-frames/"
	dest_file_name = "maximereynewcloud-" + file_prefix + "-" + prepend_zeros(str(index), 3)
	testing_density = 1/10 # 1/1 is full rendering
	nb_logs = 2
	skip_scanning = True
	only_scanning = False

	particles_textufy(source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning)
# textufy_maxime_rey_newcloud_xyzrho_frame(17, 1)

def textufy_maxime_rey_newcloud_full_168_anim():
	logger.info("Generating 168 animation frames with positions and rho...")
	
	for i in range(0, 167 + 1):
		textufy_maxime_rey_newcloud_xyzrho_frame(i + 17, i + 1)
		
	logger.success("Generated 168 animation frames.")
# textufy_maxime_rey_newcloud_full_168_anim()
