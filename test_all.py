# ANDRIX ® 2026 🤙
# 
# Test suite

# Bake imports (klodus)
from bake_eaycoberry_box import klodufy_emmaaycoberry_box
from bake_ydubois_galaxy import klodufy_yohandubois_galaxy_rho, klodufy_yohandubois_galaxy_bz

# Fry imports (particles)
from fry_ckang_galaxy import textufy_cheonsukang_bigbox_xyzrho, textufy_cheonsukang_bigbox_xyzvxvyvzrhopmetal
from fry_fthompson_starcluster import textufy_fred_thompson_starcluster_gas_xyzrho, textufy_fred_thompson_starcluster_stars_xyzmass, textufy_fred_thompson_starcluster_clusters_xyzmass
from fry_isolagal_test import textufy_isolagal_stars_xyz, textufy_isolagal_gas_xyz
from fry_jsunseri_filaments import textufy_james_sunseri_gas_xyzrho, textufy_james_sunseri_stars_xyzmass
from fry_mlombart_dustycollapse import textufy_maxime_lombart_test_collapse, textufy_maxime_lombart_zoomed_test_collapse
from fry_mrey_clouds import textufy_maxime_rey_molecularcloud_gas_xyzrho, textufy_maxime_rey_newcloud_full_168_anim, textufy_maxime_rey_newcloud_xyzrho
from fry_shan_galaxy_cluster import textufy_san_han_galaxy_cluster_xyzdensitytemp
from fry_vgoy_dustclumping import textufy_valentin_goy_test_clumping, textufy_valentin_goy_hd_test_clumping

# Klodus (bake)
klodufy_emmaaycoberry_box(True) # OK
klodufy_yohandubois_galaxy_rho(True) #OK
klodufy_yohandubois_galaxy_bz(True) #OK

# # Particles (fry) - Cheonsu Kang
# textufy_cheonsukang_bigbox_xyzrho(True) #NOK negative values log
# textufy_cheonsukang_bigbox_xyzvxvyvzrhopmetal(True) #OK

# # Particles (fry) - Fred Thompson
# textufy_fred_thompson_starcluster_gas_xyzrho(True) #OK
# textufy_fred_thompson_starcluster_stars_xyzmass(True) #OK
# textufy_fred_thompson_starcluster_clusters_xyzmass(True)

# # Particles (fry) - Isolagal
# textufy_isolagal_stars_xyz(True) #OK
# textufy_isolagal_gas_xyz(True) #OK

# # Particles (fry) - James Sunseri
# textufy_james_sunseri_gas_xyzrho(True) #OK
# textufy_james_sunseri_stars_xyzmass(True)

# # # Particles (fry) - Maxime Lombart
# textufy_maxime_lombart_test_collapse(True) #OK
# textufy_maxime_lombart_zoomed_test_collapse(True)

# # Particles (fry) - Maxime Rey
# textufy_maxime_rey_molecularcloud_gas_xyzrho(True)  #OK
# textufy_maxime_rey_newcloud_xyzrho(True)  #OK
# textufy_maxime_rey_newcloud_full_168_anim(True) #OK

# # Particles (fry) - San Han
# textufy_san_han_galaxy_cluster_xyzdensitytemp(True) #OK

# # Particles (fry) - Valentin Goy
# textufy_valentin_goy_test_clumping(True) #OK
# textufy_valentin_goy_hd_test_clumping(True)
