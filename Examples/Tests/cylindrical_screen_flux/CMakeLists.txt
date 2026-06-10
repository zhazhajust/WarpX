# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_cylindrical_screen_flux  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_cylindrical_screen_flux  # inputs
    "analysis_multidim.py --dim 2d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_cylindrical_screen_flux  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_cylindrical_screen_flux  # inputs
    "analysis_multidim.py --dim 3d"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_cylindrical_screen_flux  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_cylindrical_screen_flux  # inputs
    analysis.py  # analysis
    OFF  # checksum
    OFF  # dependency
)
