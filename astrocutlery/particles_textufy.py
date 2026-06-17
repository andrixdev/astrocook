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
from astrocutlery.utensils import round_to_n, remap, configure_loguru, is_within_box

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
		logger.error("[prepare_particles_data(...)] Unknown file type token: " + file_type_token)
		
		return False

# Main function to textufy particles data dumps, with options to customize the process
def particles_textufy (source_file, file_type_token, dest_path, dest_file_name, dimensions, kept_dimensions, minmaxs, testing_density, nb_logs, skip_scanning, only_scanning, zoombox=None):
	
	# Configure loguru
	configure_loguru()
	
	# Testing mode inits
	testing_density = min(1, testing_density) # Make sure it don't go krazy (> 1)
	testing_value = round(1/testing_density)
	
	# Load tracers data
	data = prepare_particles_data(source_file, file_type_token)
	
	# Hi
	dest_file_name = dest_file_name + ("" if testing_value == 1 else ("-1-in-" + str(testing_value)))
	logger.info("Starting work on " + dest_file_name + "...")
	
	# Prepare export file
	destination_file = open("output/" + dest_path + dest_file_name + ".txt", "w")
	
	# Get dimensions
	dims = len(dimensions)
	count = data.shape[0]
	# count = data.shape[1]
	actual_count = math.floor(count * testing_density)
	
	log_ratio = "all of " if testing_value == 1 else ("1 in " + str(testing_value) + " of all ")
	logger.info("Processing " + log_ratio + str(count) + " (== " + str(actual_count) + ") text rows to " + dest_file_name + ".txt...")
	
	step = math.floor(testing_value)
	
	# Track time taken
	start_time = datetime.datetime.now()
	
	# LOOP 1: scan
	if (not skip_scanning):
		logger.info("Scanning data to detect extrema for remapping...")
		
		# Init scanned minmax array (extremal values of positions, velocities... whatever)
		real_minmaxs = []
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
				print(str(i) + "th row is: " + row)
			
		# Log detected extrema
		for d in range(0, dims):
			dimension_name = dimensions[d][0]
			print("Min value for " + dimension_name + " is: " + str(real_minmaxs[d][0]))
			print("Max value for " + dimension_name + " is: " + str(real_minmaxs[d][1]))
			
		# Log scanning time
		mid_time = datetime.datetime.now()
		delta = mid_time.timestamp() - start_time.timestamp()
		logger.success("Scanned data in: " + str(round(delta, 2)) + " seconds.")
	
	# LOOP 2: remap & write
	is_first_line_written = True
	is_in_box = True # Init to true to avoid issues when zoombox is not set
	output_rows_count = 0
	if (not only_scanning):
		logger.info("Remapping data and writing to file...")

		if (zoombox):
			logger.warning("Filtering with zoombox: " + str(zoombox) + " (x, y, z, rad)")

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
				if (zoombox and not is_in_box):
					print(str(j) + "th remapped row is: out of zoombox")
				else:
					print(str(j) + "th remapped row is: " + row.lstrip('\n'))

			# Write to file
			if (not zoombox or (zoombox and is_in_box)):
				destination_file.write(row)
				output_rows_count += 1
				is_first_line_written = False

		# Log normalizing time
		end_time = datetime.datetime.now()
		delta = end_time.timestamp() - (mid_time.timestamp() if (not skip_scanning) else start_time.timestamp())
		logger.success("Ramapped and wrote data in: " + str(round(delta, 2)) + " seconds.")
		
		# Conclude
		logger.success("File " + dest_file_name + ".txt with " + str(output_rows_count) + " rows was created.")
