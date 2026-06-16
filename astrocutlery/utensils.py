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

def configure_loguru():
	logger.remove()
	logger.level("TRACE", color="<cyan>")
	logger.level("DEBUG", color="<blue>")
	logger.level("INFO", color="<yellow>")
	logger.level("SUCCESS", color="<green>")  # Loguru default is usually bold green
	logger.level("WARNING", color="<fg #F08>")
	logger.level("ERROR", color="<red>")      # Loguru default is usually bold red
	logger.level("CRITICAL", color="<red>")
	logger.add(sys.stderr, format="<fg #5F6>[Astrocook]</fg #5F6> <fg #2F9>[{function}]</fg #2F9> <level>[{level}]</level>: {message}")
	# logger.add(sys.stderr, level="WARNING")

	# How to use:
	# logger.trace()
	# logger.debug()
	# logger.info()
	# logger.success()
	# logger.warning()
	# logger.error()
	# logger.critical()
     