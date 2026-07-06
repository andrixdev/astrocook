import math
import sys # for loguru
from loguru import logger

def round_to_n(x, n):
    return 0 if (x == 0) else round(x, -int(math.floor(round(math.log10(abs(x)) - n + 1))))

def prepend_zeros(value, target_length):
    result = value
    size = len(str(value))
    for i in range(0, target_length - size):
        result = "0" + str(result)
        
    return result

def remap(input, source_min, source_max, target_min, target_max, clamp_mode):
    if (clamp_mode & (input < source_min)):
        return target_min
    elif (clamp_mode & (input > source_max)):
        return target_max
    else:
        return target_min + (target_max - target_min) * (input - source_min) / (source_max - source_min)
    
def is_within_box(x, y, z, x_center, y_center, z_center, radius):
	return (x >= x_center - radius) and (x <= x_center + radius) and (y >= y_center - radius) and (y <= y_center + radius) and (z >= z_center - radius) and (z <= z_center + radius)

def loguru_dynamic_formatter(record):
    
    # For trace level print raw message
    if record["level"].name == "TRACE":
        color = record["extra"].get("color", "white")
        return f"<{color}>{{message}}</{color}>\n"
    
    # For others use full format
    return "<fg #5F6>[Astrocook]</fg #5F6> <fg #2F9>[{function}]</fg #2F9> <level>[{level}]</level>: {message}\n"
     
def configure_loguru():
	logger.remove()
	logger.level("TRACE", color="<cyan>")
	logger.level("DEBUG", color="<blue>")
	logger.level("INFO", color="<yellow>")
	logger.level("SUCCESS", color="<green>")  # Loguru default is usually bold green
	logger.level("WARNING", color="<fg #F08>")
	logger.level("ERROR", color="<red>")      # Loguru default is usually bold red
	logger.level("CRITICAL", color="<red>")
    
	logger.add(sys.stderr, format=loguru_dynamic_formatter, level="TRACE")
     
	# logger.add(sys.stderr, level="WARNING")

	# How to use:
	# logger.trace()
	# logger.debug()
	# logger.info()
	# logger.success()
	# logger.warning()
	# logger.error()
	# logger.critical()
     
def get_ordinal_suffix(number: int) -> str:
    
    # Returns the appropriate ordinal suffix ('st', 'nd', 'rd', or 'th') 
    # for a given integer up to 1,000,000,000.
    
    # Enforce an upper boundary
    if not (0 <= number <= 1_000_000_000):
        logger.error("The integer must be between 0 and 1,000,000,000 inclusive.")
    
    # Extract the final two digits to check for the teen exceptions (11, 12, 13)
    last_two_digits = number % 100
    
    if last_two_digits in {11, 12, 13}:
        return "th"
    
    # Extract the absolute final digit for standard assignment
    last_digit = number % 10

    if last_digit == 1:
        return "st"
    elif last_digit == 2:
        return "nd"
    elif last_digit == 3:
        return "rd"
    else:
        return "th"
    
def update_minmaxs_of_minmaxs(minmaxs_of_minmaxs, minmaxs):

	if not minmaxs:
		return minmaxs_of_minmaxs
	
	else:
		for d in range(0, len(minmaxs_of_minmaxs)):
			# Update min value
			if (minmaxs[d][0] < minmaxs_of_minmaxs[d][0]):
				minmaxs_of_minmaxs[d][0] = minmaxs[d][0]
				
			# Update max value
			if (minmaxs[d][1] > minmaxs_of_minmaxs[d][1]):
				minmaxs_of_minmaxs[d][1] = minmaxs[d][1]

	return minmaxs_of_minmaxs

def print_minmaxs_of_minmaxs(minmaxs_of_minmaxs, dimensions):
    logger.info("Logging overall scanned minima and maxima...")
    for d in range(len(dimensions)):
        logger.bind(color="fg #DD5").trace("Overall Min value for " + str(dimensions[d][0]) + " is: " + str(minmaxs_of_minmaxs[d][0]))
        logger.bind(color="fg #DD5").trace("Overall Max value for " + str(dimensions[d][0]) + " is: " + str(minmaxs_of_minmaxs[d][1]))
