# ANDRIX ® 2025-2026 🤙
# 
# Generate text files with particle data to use in Unity for the creation of 2D textures, then sampled and rendered by a Visual Effect Graph (VFX Graph)
#
# This file reads data dumps
# It uses sarracen to read PHANTOM and SHAMROCK dumps
# It uses numpy to read NUMPY dumps
# It uses numpy to read TXT dumps
# It uses h5py to read HDF5 dumps
# It uses scipy.io to read SAV dumps

import math
import datetime
import numpy as np
from loguru import logger
from astrocutlery.utensils import round_to_n, remap, configure_loguru, is_within_box, get_ordinal_suffix

# file_type_token: "PHANTOM", "SHAMROCK", "NUMPY", "TXT", "HDF5", "HDF5-SANHAN" or "HDF5-GOY"
def prepare_particles_data(source_file, file_type_token):
	
	if (file_type_token == "PHANTOM"):
		import sarracen
		
		sdf, sdf_sinks = sarracen.read_phantom(source_file)
		
		# print(sdf.describe())
		
		return sdf
		
	elif (file_type_token == "SHAMROCK"):
		sdf = sarracen.read_shamrock(source_file)
		
		# print(sdf.describe())
		
		return sdf

	elif (file_type_token == "NUMPY"):
		data = np.load(source_file)
		
		logger.info("Data shape is " + str(data.shape) + " with a total of " + str(data.size) + " elements.")
		
		return data
		
	elif (file_type_token == "TXT"):
		data = arr = np.loadtxt(source_file)
		
		logger.info("Data shape is " + str(data.shape) + " with a total of " + str(data.size) + " elements.")
		
		return data
		
	elif (file_type_token == "HDF5" or file_type_token == "HDF5-SANHAN"):
		import h5py
		
		with h5py.File(source_file, "r") as f:
			# List all keys
			if (file_type_token == "HDF5-SANHAN"):
				file = f["data"]
			elif (file_type_token == "HDF5"):
				file = f

			keys = list(file.keys())

			# Reorder keys to have x, y and z first if they exist
			for dim in ["z", "y", "x"]:
				if dim in keys:
					keys.remove(dim)
					keys.insert(0, dim)

			logger.info("HDF5 keys: %s" % keys)
			
			# Load all datasets and stack them
			datasets = [np.array(file[key]) for key in keys]
			data = np.column_stack(datasets) if len(datasets) > 1 else np.array(datasets[0])
			
			logger.info("Data shape is " + str(data.shape) + " with a total of " + str(data.size) + " elements.")
			
			return data

	elif (file_type_token == "HDF5-GOY"):
		import h5py
		
		with h5py.File(source_file, "r") as hdf:
			# Logging HDF5 structure
			logger.info("Printing Valentin Goy HDF5 file structure...")
			items = []
			hdf.visit(items.append)
			logger.info(f"Discovered HDF5 structure: {', '.join(items)}")

			# Read Alex group's columns
			x = hdf['Alex/x'][:]
			rho = hdf['Alex/rho'][:]
			rhod = hdf['Alex/rhod'][:]
			vx = hdf['Alex/vx'][:]
			vy = hdf['Alex/vy'][:]
			vz = hdf['Alex/vz'][:]
			vdx = hdf['Alex/vdx'][:]
			vdy = hdf['Alex/vdy'][:]
			vdz = hdf['Alex/vdz'][:]
			Bx = hdf['Alex/Bx'][:]
			By = hdf['Alex/By'][:]
			Bz = hdf['Alex/Bz'][:]
			
			columns = [x, rho, rhod, vx, vy, vz, vdx, vdy, vdz, Bx, By, Bz]
			
			# Optionally create sd column if available
			if 'Alex/sd' in hdf:
				sd = hdf['Alex/sd'][:]
				columns.append(sd)
			
			# Stack into numpy array
			data = np.column_stack(columns)

			logger.info("Loaded Valentin Goy HDF5 data with shape: " + str(data.shape) + " and a total of " + str(data.size) + " elements.")

			return data

	elif (file_type_token == "SAV"):
		from scipy.io import readsav
		
		sav = readsav(source_file)
		print(sav.keys())
		cell = sav.cell
		x, y, z, dx = cell.x[0], cell.y[0], cell.z[0], cell.dx[0]
		variables = cell[0][4]
		rho = variables[0]
		vx = variables[1]
		vy = variables[2]
		vz = variables[3]
		pressure = variables[4]
		metallicity = variables[5]

		data = np.column_stack([x, y, z, vx, vy, vz, rho, pressure, metallicity])

		logger.info("Data shape is " + str(data.shape) + " with a total of " + str(data.size) + " elements.")
			
		return data

		# #cell center position and size
		# x, y, z, dx = cell.x[0], cell.y[0], cell.z[0], cell.dx[0]
		# var  = cell[0][4]
		# #hydro variables, density, vx, vy, vz, pressure, metallicity, ...
		# d = var[0]; p = var[4]

	else:
		logger.error("Unknown file type token: " + file_type_token)
		
		return False

def compute_loop_variables(data, testing_density):
	testing_value = round(1/testing_density)

	# Get step
	step = math.floor(testing_value)

	# Get actual_count
	count = data.shape[0]
	actual_count = math.floor(count * testing_density)
	log_ratio = "all of " if testing_value == 1 else ("1 in " + str(testing_value) + " of all ")
	logger.info("About to process " + log_ratio + str(count) + " (== " + str(actual_count) + ") text rows...")

	return [step, actual_count]

def enrich_output_file_name(dest_file_name, testing_density):
	testing_value = round(1/testing_density)

	return dest_file_name + ("" if testing_value == 1 else ("-1-in-" + str(testing_value)))
	
def particles_scan(data, actual_count, step, dimensions, file_type_token, nb_logs):
	logger.info("Scanning data to detect extrema for remapping...")
	start_time = datetime.datetime.now()
	
	# Init scanned minmax array (extremal values of positions, velocities... whatever)
	real_minmaxs = []
	dims = len(dimensions)
	for d in range(0, dims):
		real_minmaxs.append([float("inf"), float("-inf")])
	
	# Start loop
	ii = 0
	for i in range(0, actual_count):
		ii = i * step
		
		row = ""
		
		for d in range(0, dims):
			dimension_name = dimensions[d][0]
			dimension_mode = dimensions[d][1]
			
			# Grab data value Shamrock/Phantom way (dimension name)
			if (file_type_token == "SHAMROCK"):
				# Special case for Yona's rho, derived from hpart
				if (dimension_name == "rho"):
					val = 1 * (data.iloc[ii]["hpart"] ** 3)
				else:
					val = data.iloc[ii][dimension_name]
			
			elif (file_type_token == "PHANTOM"):
				val = data.iloc[ii][dimension_name]
				
			# Grab data value basic way (just the order)
			elif (file_type_token == "NUMPY" or file_type_token == "TXT" or file_type_token == "HDF5" or file_type_token == "HDF5-SANHAN" or file_type_token == "SAV" or file_type_token == "HDF5-GOY"):
				val = data[ii][d]
				
			# Checking mode
			if (dimension_mode == "log"):
				val = math.log10(val)
				
			# Rounding (5 digits just for the scan)
			val = round_to_n(val, 5)
			
			# Feed row to potentially print
			if (d > 0):
				row = row + " "
			row = row + str(val)
			
			# Update max value
			if (val > real_minmaxs[d][1]):
				real_minmaxs[d][1] = val
				
			# Update min value
			if (val < real_minmaxs[d][0]):
				real_minmaxs[d][0] = val
			
		if (i % max(1, int(round(actual_count/nb_logs))) == 0):
			logger.bind(color="fg #E44").trace(str(i) + "th row is: " + row)
		
	# Log detected extrema
	for d in range(0, dims):
		dimension_name = dimensions[d][0]
		logger.bind(color="fg #DA7").trace("Min value for " + dimension_name + " is: " + str(real_minmaxs[d][0]))
		logger.bind(color="fg #DA7").trace("Max value for " + dimension_name + " is: " + str(real_minmaxs[d][1]))
		
	# Log scanning time
	end_time = datetime.datetime.now()
	delta = end_time.timestamp() - start_time.timestamp()
	logger.success("Scanned data in: " + str(round(delta, 2)) + " seconds.")

def particles_export(data, actual_count, step, dimensions, kept_dimensions, minmaxs, file_type_token, nb_logs, dest_path, dest_file_name, zoombox=False):
	# Log start info
	logger.info("Exporting: remapping data and writing to file...")
	start_time = datetime.datetime.now()
	if (zoombox):
		logger.warning("Filtering with zoombox: " + str(zoombox) + " (x, y, z, rad)")

	# Open export destination file
	destination_file = open("output/" + dest_path + dest_file_name + ".txt", "w")

	# Init
	is_first_line_written = True
	is_in_box = True # Init to true to avoid issues when zoombox is not set
	output_rows_count = 0

	# Loop
	dims = len(dimensions)
	for j in range(0, actual_count):
		jj = j * step
		
		row = ""
		
		# Prepare remap
		low_quality_digits = 3
		high_quality_digits = 6
		lq_max = 10 ** low_quality_digits
		hq_max = 10 ** high_quality_digits
		
		# Prepare boolean to check if inside the zoomed box
		if (zoombox):
			is_in_box = is_within_box(data[jj][0], data[jj][1], data[jj][2], zoombox[0], zoombox[1], zoombox[2], zoombox[3])

		if (not is_first_line_written):
			row += "\n"

		for d in range(0, dims):
			dimension_name = dimensions[d][0]
			dimension_mode = dimensions[d][1]
			dimension_quality = dimensions[d][2]
			is_dimension_kept = True if (kept_dimensions[d] == 1) else False
			
			digits = low_quality_digits if (dimension_quality == "LQ") else high_quality_digits
			
			# Grab data value Shamrock/Phantom way (dimension name)
			if (file_type_token == "SHAMROCK"):
				# Special case for Yona's rho, derived from hpart
				if (dimension_name == "rho"):
					val = 1 * (data.iloc[jj]["hpart"] ** 3)
				else:
					val = data.iloc[jj][dimension_name]                   

			elif (file_type_token == "PHANTOM"):
				val = data.iloc[jj][dimension_name]
				
			# Grab data value basic way (just the order)
			elif (file_type_token == "NUMPY" or file_type_token == "TXT" or file_type_token == "HDF5" or file_type_token == "HDF5-SANHAN" or file_type_token == "SAV" or file_type_token == "HDF5-GOY"):
				val = data[jj][d]

			# Checking mode
			if (dimension_mode == "log"):
				val = math.log10(val)
			
			# Remap
			min_val = minmaxs[d][0]
			max_val = minmaxs[d][1]
			min_target = 0
			max_target = lq_max if (dimension_quality == "LQ") else hq_max
			val = int(round_to_n(remap(val, min_val, max_val, min_target, max_target, True), digits + 1))
			
			# Feed row to later write to file
			if (is_dimension_kept):
				if (d > 0):
					row = row + " "
				
				if (not zoombox or (zoombox and is_in_box)):
					row = row + str(val)
					
		# Log row sometimes
		if (j % max(1, int(round(actual_count/nb_logs))) == 0):
			th_nb = str(j + 1) + get_ordinal_suffix(j + 1)
			content = ""
			
			if (zoombox and not is_in_box):
				content = "out of zoombox"
			else:
				content = row.lstrip('\n')
			
			logger.bind(color="fg #3FB").trace(th_nb + " remapped row is: " + content)

		# Write to file
		if (not zoombox or (zoombox and is_in_box)):
			destination_file.write(row)
			output_rows_count += 1
			is_first_line_written = False

	# Log export time
	end_time = datetime.datetime.now()
	delta = end_time.timestamp() - start_time.timestamp()
	logger.success("Exported data in: " + str(round(delta, 2)) + " seconds.")

	# Conclude
	logger.success("File " + dest_file_name + ".txt with " + str(output_rows_count) + " rows was created.")


# Main function to textufy particles data dumps, with options to customize the process
def particles_textufy (source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, is_scanning, is_exporting, zoombox=None):

	# Configure loguru
	configure_loguru()

	# Check if the method should run anything
	if (not is_scanning and not is_exporting):
		logger.error("Neither scanning nor exporting, aborting function.")
		return

	# Secure input arguments
	testing_density = min(1, testing_density)

	# Load particles data
	data = prepare_particles_data(source_file, file_type_token)

	# Prepare output file name
	dest_file_name = enrich_output_file_name(dest_file_name, testing_density)
	logger.info("Starting work on " + dest_file_name + "...")
	
	# Get loop variables (step and acutal_count)
	loop_vars = compute_loop_variables(data, testing_density)
	step = loop_vars[0]
	actual_count = loop_vars[1]
	
	# LOOP 1: scan
	if (is_scanning):
		particles_scan(data, actual_count, step, dimensions, file_type_token, nb_logs)
	
	# LOOP 2: export (remap & write)
	if (is_exporting):
		particles_export(data, actual_count, step, dimensions, kept_dimensions, minmaxs, file_type_token, nb_logs, dest_path, dest_file_name, zoombox=False)
