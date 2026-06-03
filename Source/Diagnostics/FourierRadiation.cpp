#include "FourierRadiation.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace amrex;

FourierRadiation::FourierRadiation ()
{
    ParmParse const pp_warpx("warpx");
    pp_warpx.query("do_fourier_radiation", m_enabled);
    if (!m_enabled) {
        return;
    }

#if !defined(WARPX_DIM_3D)
    WARPX_ABORT_WITH_MESSAGE("Inline Fourier radiation is currently implemented only for 3D.");
#endif

    pp_warpx.get("fourier_radiation_species", m_species_name);

    Vector<Real> omega_params;
    Vector<Real> theta_params;
    Vector<Real> phi_params;
    utils::parser::getArrWithParser(pp_warpx, "fourier_radiation_omega", omega_params);
    utils::parser::getArrWithParser(pp_warpx, "fourier_radiation_theta", theta_params);
    utils::parser::getArrWithParser(pp_warpx, "fourier_radiation_phi", phi_params);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        omega_params.size() == 3 && theta_params.size() == 3 && phi_params.size() == 3,
        "warpx.fourier_radiation_omega/theta/phi must each be: min max num_points.");

    m_omega_min = omega_params[0];
    m_omega_max = omega_params[1];
    m_num_omega = static_cast<int>(std::llround(omega_params[2]));
    m_theta_min = theta_params[0];
    m_theta_max = theta_params[1];
    m_num_theta = static_cast<int>(std::llround(theta_params[2]));
    m_phi_min = phi_params[0];
    m_phi_max = phi_params[1];
    m_num_phi = static_cast<int>(std::llround(phi_params[2]));
    pp_warpx.query("fourier_radiation_omega_grid", m_omega_grid);
    utils::parser::queryWithParser(
        pp_warpx, "fourier_radiation_min_particle_energy", m_min_particle_energy);
    utils::parser::queryWithParser(
        pp_warpx, "fourier_radiation_particle_fraction", m_particle_fraction);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_num_omega > 0 && m_num_theta > 0 && m_num_phi > 0,
        "Fourier radiation grid dimensions must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_omega_min > 0._rt && m_omega_max > 0._rt,
        "Fourier radiation omega bounds must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_omega_grid == "linear" || m_omega_grid == "log",
        "warpx.fourier_radiation_omega_grid must be either 'linear' or 'log'.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_min_particle_energy >= 0._rt,
        "warpx.fourier_radiation_min_particle_energy must be non-negative.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_particle_fraction > 0._rt && m_particle_fraction <= 1._rt,
        "warpx.fourier_radiation_particle_fraction must be in the interval (0, 1].");

    pp_warpx.query("fourier_radiation_reset_after_output", m_reset_after_output);

    Allocate();
}

bool
FourierRadiation::IsSpeciesSelected (std::string const& species_name) const
{
    return m_enabled && species_name == m_species_name;
}

void
FourierRadiation::Allocate ()
{
    m_num_grid_nodes = m_num_omega * m_num_theta * m_num_phi;

    Vector<Real> h_omega(m_num_omega);
    Vector<Real> h_sin_theta(m_num_theta);
    Vector<Real> h_cos_theta(m_num_theta);
    Vector<Real> h_sin_phi(m_num_phi);
    Vector<Real> h_cos_phi(m_num_phi);
    m_frequency.resize(m_num_omega);

    for (int i = 0; i < m_num_omega; ++i) {
        Real const frac = (m_num_omega == 1) ? 0._rt : Real(i) / Real(m_num_omega - 1);
        if (m_omega_grid == "log") {
            m_frequency[i] = std::exp(
                std::log(m_omega_min) + frac * (std::log(m_omega_max) - std::log(m_omega_min)));
        } else {
            m_frequency[i] = m_omega_min + frac * (m_omega_max - m_omega_min);
        }
        h_omega[i] = 2._rt * Math::pi<Real>() * m_frequency[i];
    }
    for (int i = 0; i < m_num_theta; ++i) {
        Real const frac = (m_num_theta == 1) ? 0._rt : Real(i) / Real(m_num_theta - 1);
        Real const theta = m_theta_min + frac * (m_theta_max - m_theta_min);
        h_sin_theta[i] = std::sin(theta);
        h_cos_theta[i] = std::cos(theta);
    }
    for (int i = 0; i < m_num_phi; ++i) {
        Real const phi = m_phi_min + (m_phi_max - m_phi_min) / Real(m_num_phi) * Real(i);
        h_sin_phi[i] = std::sin(phi);
        h_cos_phi[i] = std::cos(phi);
    }

    m_omega_2pi.resize(m_num_omega);
    m_sin_theta.resize(m_num_theta);
    m_cos_theta.resize(m_num_theta);
    m_sin_phi.resize(m_num_phi);
    m_cos_phi.resize(m_num_phi);
    Gpu::copy(Gpu::hostToDevice, h_omega.begin(), h_omega.end(), m_omega_2pi.begin());
    Gpu::copy(Gpu::hostToDevice, h_sin_theta.begin(), h_sin_theta.end(), m_sin_theta.begin());
    Gpu::copy(Gpu::hostToDevice, h_cos_theta.begin(), h_cos_theta.end(), m_cos_theta.begin());
    Gpu::copy(Gpu::hostToDevice, h_sin_phi.begin(), h_sin_phi.end(), m_sin_phi.begin());
    Gpu::copy(Gpu::hostToDevice, h_cos_phi.begin(), h_cos_phi.end(), m_cos_phi.begin());

    m_amp_x_re.resize(m_num_grid_nodes, 0._rt);
    m_amp_x_im.resize(m_num_grid_nodes, 0._rt);
    m_amp_y_re.resize(m_num_grid_nodes, 0._rt);
    m_amp_y_im.resize(m_num_grid_nodes, 0._rt);
    m_amp_z_re.resize(m_num_grid_nodes, 0._rt);
    m_amp_z_im.resize(m_num_grid_nodes, 0._rt);
}

void
FourierRadiation::GetIntensity (std::vector<Real>& intensity) const
{
    intensity.assign(m_num_grid_nodes, 0._rt);
    if (!m_enabled) {
        return;
    }

    Vector<Real> axr(m_num_grid_nodes);
    Vector<Real> axi(m_num_grid_nodes);
    Vector<Real> ayr(m_num_grid_nodes);
    Vector<Real> ayi(m_num_grid_nodes);
    Vector<Real> azr(m_num_grid_nodes);
    Vector<Real> azi(m_num_grid_nodes);

    Gpu::copy(Gpu::deviceToHost, m_amp_x_re.begin(), m_amp_x_re.end(), axr.begin());
    Gpu::copy(Gpu::deviceToHost, m_amp_x_im.begin(), m_amp_x_im.end(), axi.begin());
    Gpu::copy(Gpu::deviceToHost, m_amp_y_re.begin(), m_amp_y_re.end(), ayr.begin());
    Gpu::copy(Gpu::deviceToHost, m_amp_y_im.begin(), m_amp_y_im.end(), ayi.begin());
    Gpu::copy(Gpu::deviceToHost, m_amp_z_re.begin(), m_amp_z_re.end(), azr.begin());
    Gpu::copy(Gpu::deviceToHost, m_amp_z_im.begin(), m_amp_z_im.end(), azi.begin());

    ParallelDescriptor::ReduceRealSum(axr.data(), static_cast<int>(axr.size()));
    ParallelDescriptor::ReduceRealSum(axi.data(), static_cast<int>(axi.size()));
    ParallelDescriptor::ReduceRealSum(ayr.data(), static_cast<int>(ayr.size()));
    ParallelDescriptor::ReduceRealSum(ayi.data(), static_cast<int>(ayi.size()));
    ParallelDescriptor::ReduceRealSum(azr.data(), static_cast<int>(azr.size()));
    ParallelDescriptor::ReduceRealSum(azi.data(), static_cast<int>(azi.size()));

    for (int i = 0; i < m_num_grid_nodes; ++i) {
        constexpr Real prefactor =
            1._rt / (16._rt * Math::pi<Real>() * Math::pi<Real>() * Math::pi<Real>()
                     * PhysConst::epsilon_0 * PhysConst::c);
        intensity[i] = prefactor * (axr[i]*axr[i] + axi[i]*axi[i]
                                  + ayr[i]*ayr[i] + ayi[i]*ayi[i]
                                  + azr[i]*azr[i] + azi[i]*azi[i]);
    }
}

void
FourierRadiation::Reset ()
{
    if (!m_enabled) {
        return;
    }

    Real* const amp_x_re = m_amp_x_re.dataPtr();
    Real* const amp_x_im = m_amp_x_im.dataPtr();
    Real* const amp_y_re = m_amp_y_re.dataPtr();
    Real* const amp_y_im = m_amp_y_im.dataPtr();
    Real* const amp_z_re = m_amp_z_re.dataPtr();
    Real* const amp_z_im = m_amp_z_im.dataPtr();
    int const n = m_num_grid_nodes;

    ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
    {
        amp_x_re[i] = 0._rt;
        amp_x_im[i] = 0._rt;
        amp_y_re[i] = 0._rt;
        amp_y_im[i] = 0._rt;
        amp_z_re[i] = 0._rt;
        amp_z_im[i] = 0._rt;
    });
}
