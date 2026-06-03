# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_fourier_radiation_accelerated  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_fourier_radiation_accelerated  # inputs
    "analysis_fourier_radiation.py accelerated"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_fourier_radiation_energy_spread_beam  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_fourier_radiation_energy_spread_beam  # inputs
    "analysis_fourier_radiation.py energy_spread_beam"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_fourier_radiation_filtered  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_fourier_radiation_filtered  # inputs
    "analysis_fourier_radiation.py filtered"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_fourier_radiation_undulator  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_fourier_radiation_undulator  # inputs
    "analysis_fourier_radiation.py undulator"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_fourier_radiation_uniform  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_fourier_radiation_uniform  # inputs
    "analysis_fourier_radiation.py uniform"  # analysis
    OFF  # checksum
    OFF  # dependency
)
