/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_CYLINDRICALSCREENFLUX_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_CYLINDRICALSCREENFLUX_H_

#include "ReducedDiags.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <cstdint>
#include <string>
#include <vector>

/**
 * Records particles that cross a cylindrical screen while moving inward.
 *
 * In RZ and 3D, the screen is a cylinder. In 2D XZ, it is the fixed-theta
 * cross-section of that cylinder: one or two planar screens at x = x0 +/- r0.
 */
class CylindricalScreenFlux : public ReducedDiags {
  public:
    /**
     * constructor
     * @param[in] rd_name reduced diagnostics name
     */
    explicit CylindricalScreenFlux (const std::string& rd_name);

    /**
     * Record crossing events and update previous particle radii.
     *
     * @param[in] step current time step
     */
    void ComputeDiags (int step) final;

    /**
     * Event rows are written from ComputeDiags on each rank.
     *
     * @param[in] step current time step
     */
    void WriteToFile (int step) const final;

  private:
    struct SpeciesState {
        int m_index = -1;
        std::string m_name;
        std::string m_previous_radius_name;
    };

    void WriteParticleHeader () const;
    void WriteHistogramHeader () const;
    void WriteHistogram (
        int step,
        amrex::Real time,
        int interval_index,
        amrex::Real interval_start,
        amrex::Real interval_stop,
        std::vector<amrex::Real> const& histogram) const;

    [[nodiscard]] std::size_t HistogramIndex (
        int species_index, int theta_bin, int z_bin, int energy_bin, int component) const;

    void ResetHistogram ();

    amrex::ParticleReal m_r0 = 0.0;
    amrex::ParticleReal m_center_x = 0.0;
    amrex::ParticleReal m_center_y = 0.0;
    amrex::ParticleReal m_z_min = 0.0;
    amrex::ParticleReal m_z_max = 0.0;
    amrex::ParticleReal m_theta_2d = 0.0;
    std::string m_file_name;
    std::string m_histogram_file_name;
    bool m_write_particles = true;
    bool m_histogram_enabled = false;
    bool m_write_histogram_header = false;
    bool m_two_sided_2d = true;
    bool m_has_previous_radius = false;
    int m_bins_theta = 0;
    int m_bins_z = 0;
    int m_bins_energy = 0;
    int m_histogram_interval_index = 0;
    amrex::Real m_time_interval = 0.0;
    amrex::Real m_interval_start = 0.0;
    amrex::Real m_interval_stop = 0.0;
    amrex::ParticleReal m_energy_min = 0.0;
    amrex::ParticleReal m_energy_max = 0.0;
    std::vector<SpeciesState> m_species;
    std::vector<amrex::Real> m_histogram;
    amrex::Gpu::DeviceVector<amrex::Real> m_histogram_device;
};

#endif
