/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "TimeDomainRadiation.H"

#include "Utils/TextMsg.H"

#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>

using namespace amrex;

TimeDomainRadiation::TimeDomainRadiation ()
{
    const ParmParse pp_warpx("warpx");
    Vector<std::string> rd_names;
    pp_warpx.queryarr("reduced_diags_names", rd_names);

    int num_time_domain_radiation_diags = 0;
    std::string diag_name;
    for (auto const& rd_name : rd_names) {
        const ParmParse pp_rd(rd_name);
        std::string rd_type;
        if (pp_rd.query("type", rd_type) && rd_type == "TimeDomainRadiation") {
            ++num_time_domain_radiation_diags;
            diag_name = rd_name;
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_time_domain_radiation_diags <= 1,
        "Only one TimeDomainRadiation reduced diagnostic is currently supported.");

    if (num_time_domain_radiation_diags == 0) {
        return;
    }

#if !defined(WARPX_DIM_3D)
    WARPX_ABORT_WITH_MESSAGE("TimeDomainRadiation is currently implemented only for 3D.");
#endif

    m_enabled = true;
    const ParmParse pp_rd(diag_name);
    pp_rd.getarr("species", m_species_names);
    pp_rd.get("n_time", m_n_time);
    pp_rd.get("n_theta", m_n_theta);
    pp_rd.get("n_phi", m_n_phi);
    pp_rd.get("time_min", m_time_min);
    pp_rd.get("time_max", m_time_max);
    pp_rd.get("theta_min", m_theta_min);
    pp_rd.get("theta_max", m_theta_max);
    pp_rd.get("phi_min", m_phi_min);
    pp_rd.get("phi_max", m_phi_max);
    pp_rd.get("radius", m_radius);
    pp_rd.query("far_field_approx", m_far_field_approx);
    pp_rd.query("reset_after_output", m_reset_after_output);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_species_names.empty(),
        "TimeDomainRadiation requires at least one species.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_time > 1 && m_n_theta > 0 && m_n_phi > 0,
        "TimeDomainRadiation grid dimensions must satisfy n_time > 1, n_theta > 0, n_phi > 0.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_time_max > m_time_min,
        "TimeDomainRadiation requires time_max > time_min.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_radius > amrex::Real(0),
        "TimeDomainRadiation radius must be positive.");

    m_dt_screen = (m_time_max - m_time_min) / (m_n_time - 1);
    m_dtheta = (m_n_theta > 1)
        ? (m_theta_max - m_theta_min) / (m_n_theta - 1) : amrex::Real(0);
    m_dphi = (m_n_phi > 1)
        ? (m_phi_max - m_phi_min) / (m_n_phi - 1) : amrex::Real(0);
    m_num_grid_nodes = m_n_time * m_n_theta * m_n_phi;

    Allocate();
}

bool
TimeDomainRadiation::IsSpeciesSelected (std::string const& species_name) const
{
    return m_enabled &&
        std::find(m_species_names.begin(), m_species_names.end(), species_name) != m_species_names.end();
}

void
TimeDomainRadiation::Allocate ()
{
    m_field.resize(static_cast<std::size_t>(m_n_components) * m_num_grid_nodes);
    m_sin_theta.resize(m_n_theta);
    m_cos_theta.resize(m_n_theta);
    m_sin_phi.resize(m_n_phi);
    m_cos_phi.resize(m_n_phi);

    Vector<Real> h_sin_theta(m_n_theta);
    Vector<Real> h_cos_theta(m_n_theta);
    Vector<Real> h_sin_phi(m_n_phi);
    Vector<Real> h_cos_phi(m_n_phi);

    for (int i = 0; i < m_n_theta; ++i) {
        Real const theta = Theta(i);
        h_sin_theta[i] = std::sin(theta);
        h_cos_theta[i] = std::cos(theta);
    }
    for (int i = 0; i < m_n_phi; ++i) {
        Real const phi = Phi(i);
        h_sin_phi[i] = std::sin(phi);
        h_cos_phi[i] = std::cos(phi);
    }

    Gpu::copy(Gpu::hostToDevice, h_sin_theta.begin(), h_sin_theta.end(), m_sin_theta.begin());
    Gpu::copy(Gpu::hostToDevice, h_cos_theta.begin(), h_cos_theta.end(), m_cos_theta.begin());
    Gpu::copy(Gpu::hostToDevice, h_sin_phi.begin(), h_sin_phi.end(), m_sin_phi.begin());
    Gpu::copy(Gpu::hostToDevice, h_cos_phi.begin(), h_cos_phi.end(), m_cos_phi.begin());
    Reset();
}

TimeDomainRadiationDeviceData
TimeDomainRadiation::GetDeviceData ()
{
    return TimeDomainRadiationDeviceData{
        m_field.data(), m_sin_theta.data(), m_cos_theta.data(), m_sin_phi.data(),
        m_cos_phi.data(), m_n_time, m_n_theta, m_n_phi, m_n_components, m_time_min,
        m_time_max, m_dt_screen, m_radius, m_far_field_approx};
}

void
TimeDomainRadiation::AddToHostVector (std::vector<Real>& data) const
{
    data.resize(m_field.size());
    Gpu::copy(Gpu::deviceToHost, m_field.begin(), m_field.end(), data.begin());
    Gpu::synchronize();
    ParallelDescriptor::ReduceRealSum(data.data(), static_cast<int>(data.size()));

    for (auto& value : data) {
        value /= m_dt_screen;
    }
}

void
TimeDomainRadiation::Reset ()
{
    Gpu::fillAsync(m_field.begin(), m_field.end(),
        [] AMREX_GPU_DEVICE (amrex::Real& value, long) noexcept { value = amrex::Real(0); });
    Gpu::synchronize();
}
