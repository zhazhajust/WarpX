# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_time_domain_radiation  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_time_domain_radiation  # inputs
    "analysis_time_domain_radiation.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_time_domain_radiation_undulator  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_time_domain_radiation_undulator  # inputs
    "analysis_time_domain_radiation.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
