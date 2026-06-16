/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_TIMEDOMAINRADIATION_H_
#define WARPX_DIAGNOSTICS_TIMEDOMAINRADIATION_H_

#include "Diagnostics/TimeDomainRadiation_K.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <memory>
#include <string>
#include <vector>

class TimeDomainRadiation
{
public:
    TimeDomainRadiation ();

    [[nodiscard]] bool Enabled () const { return m_enabled; }
    [[nodiscard]] bool IsSpeciesSelected (std::string const& species_name) const;
    [[nodiscard]] bool ResetAfterOutput () const { return m_reset_after_output; }

    [[nodiscard]] int NumTime () const { return m_n_time; }
    [[nodiscard]] int NumTheta () const { return m_n_theta; }
    [[nodiscard]] int NumPhi () const { return m_n_phi; }
    [[nodiscard]] int NumComponents () const { return m_n_components; }
    [[nodiscard]] int NumGridNodes () const { return m_num_grid_nodes; }
    [[nodiscard]] amrex::Real FilterRandomFraction () const { return m_filter_random_fraction; }
    [[nodiscard]] int FilterUnistride () const { return m_filter_unistride; }
    [[nodiscard]] bool FilterDoFunction () const { return m_filter_do_function; }
    [[nodiscard]] amrex::Parser const* FilterParser () const { return m_filter_parser.get(); }
    [[nodiscard]] bool DoParticleFilter () const {
        return m_filter_random_fraction < amrex::Real(1) ||
            m_filter_unistride > 1 || m_filter_do_function;
    }

    [[nodiscard]] amrex::Real Time (int i) const { return m_time_min + i*m_dt_screen; }
    [[nodiscard]] amrex::Real Theta (int i) const { return m_theta_min + i*m_dtheta; }
    [[nodiscard]] amrex::Real Phi (int i) const { return m_phi_min + i*m_dphi; }

    [[nodiscard]] TimeDomainRadiationDeviceData GetDeviceData ();

    void AddToHostVector (std::vector<amrex::Real>& data) const;
    void Reset ();

private:
    void Allocate ();

    bool m_enabled = false;
    bool m_far_field_approx = true;
    bool m_reset_after_output = false;
    int m_n_time = 0;
    int m_n_theta = 0;
    int m_n_phi = 0;
    int m_n_components = 3;
    int m_num_grid_nodes = 0;
    amrex::Real m_time_min = amrex::Real(0);
    amrex::Real m_time_max = amrex::Real(0);
    amrex::Real m_theta_min = amrex::Real(0);
    amrex::Real m_theta_max = amrex::Real(0);
    amrex::Real m_phi_min = amrex::Real(0);
    amrex::Real m_phi_max = amrex::Real(0);
    amrex::Real m_radius = amrex::Real(1);
    amrex::Real m_dt_screen = amrex::Real(1);
    amrex::Real m_dtheta = amrex::Real(1);
    amrex::Real m_dphi = amrex::Real(1);
    amrex::Real m_filter_random_fraction = amrex::Real(1);
    int m_filter_unistride = 1;
    bool m_filter_do_function = false;
    std::unique_ptr<amrex::Parser> m_filter_parser;
    amrex::Vector<std::string> m_species_names;

    amrex::Gpu::DeviceVector<amrex::Real> m_field;
    amrex::Gpu::DeviceVector<amrex::Real> m_sin_theta;
    amrex::Gpu::DeviceVector<amrex::Real> m_cos_theta;
    amrex::Gpu::DeviceVector<amrex::Real> m_sin_phi;
    amrex::Gpu::DeviceVector<amrex::Real> m_cos_phi;
};

#endif
