# ANDRIX 2025-2026
#
# Bakes the klodus using cube_klodufy

from astrocutlery.cube_klodufy import klodufy
from astrocutlery.utensils import prepend_zeros


# YOUNGDISKDOLLS (seven density cubes from one snapshot)
def klodufy_youngdiskdolls_rho(is_test=False):
    dimensions = [["rho", "log"]]
    file_type_token = "NUMPY"
    quality = "low"
    dest_path = "youngdiskdolls/1-frame/"
    testing_density = 1/1 if not is_test else 1/10
    nb_logs = 20

    for cube_index in range(7):
        source_file = (
            "./input/youngdiskdolls/1-frame/russian_doll_"
            + prepend_zeros(cube_index, 5)
            + ".npy"
        )
        size = 512
        minmaxs = []
        dest_file_name = ""
        is_scanning = True
        is_exporting = False

        scan = klodufy(source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, is_scanning, is_exporting)

        minmaxs = scan[0]
        size = scan[1]
        dest_file_name = (
            "youngdiskdolls-rho-"
            + str(size)
            + "-"
            + prepend_zeros(cube_index + 1, 3)
            + ("-testing" if is_test else "")
        )
        print("Cube", cube_index, "minmaxs:", minmaxs)
        is_scanning = False
        is_exporting = True

        klodufy(source_file, file_type_token, size, dimensions, minmaxs, quality, dest_path, dest_file_name, testing_density, nb_logs, is_scanning, is_exporting)

if __name__ == "__main__":
    is_test = False
    klodufy_youngdiskdolls_rho(is_test)
