# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_cylindrical_screen_flux  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_cylindrical_screen_flux  # inputs
    analysis.py  # analysis
    OFF  # checksum
    OFF  # dependency
)
