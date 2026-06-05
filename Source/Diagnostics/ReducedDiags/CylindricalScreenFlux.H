/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_CYLINDRICALSCREENFLUX_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_CYLINDRICALSCREENFLUX_H_

#include "ReducedDiags.H"

#include <AMReX_REAL.H>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

/**
 * Records particles that cross a cylindrical screen in RZ while moving inward.
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
        std::map<std::uint64_t, amrex::ParticleReal> m_previous_r;
    };

    amrex::ParticleReal m_r0 = 0.0;
    amrex::ParticleReal m_z_min = 0.0;
    amrex::ParticleReal m_z_max = 0.0;
    std::string m_file_name;
    std::vector<SpeciesState> m_species;
};

#endif
