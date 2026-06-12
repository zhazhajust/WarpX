#include "FourierRadiation.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>

using namespace amrex;
namespace fr = warpx::diagnostics::fourier_radiation;

FourierRadiation::FourierRadiation ()
{
    ParmParse const pp_warpx("warpx");
    Vector<std::string> rd_names;
    pp_warpx.queryarr("reduced_diags_names", rd_names);

    std::string rd_name;
    int num_fourier_radiation_diags = 0;
    for (auto const& name : rd_names) {
        ParmParse const pp_rd(name);
        std::string rd_type;
        if (pp_rd.query("type", rd_type) && rd_type == "FourierRadiation") {
            rd_name = name;
            ++num_fourier_radiation_diags;
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_fourier_radiation_diags <= 1,
        "Only one FourierRadiation reduced diagnostic is currently supported.");

    if (num_fourier_radiation_diags == 0) {
        return;
    }
    m_enabled = true;

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("Inline Fourier radiation is currently implemented only for 3D and RZ.");
#endif

    ParmParse const pp_fr(rd_name);
    pp_fr.get("species", m_species_name);

    Vector<Real> frequency_params;
    Vector<Real> theta_params;
    Vector<Real> phi_params;
    utils::parser::getArrWithParser(pp_fr, "frequencies", frequency_params);
    utils::parser::getArrWithParser(pp_fr, "theta", theta_params);
    utils::parser::getArrWithParser(pp_fr, "phi", phi_params);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        frequency_params.size() == 3 && theta_params.size() == 3 && phi_params.size() == 3,
        rd_name + ".frequencies/theta/phi must each be: min max num_points.");

    m_omega_min = frequency_params[0];
    m_omega_max = frequency_params[1];
    m_num_omega = static_cast<int>(std::llround(frequency_params[2]));
    m_theta_min = theta_params[0];
    m_theta_max = theta_params[1];
    m_num_theta = static_cast<int>(std::llround(theta_params[2]));
    m_phi_min = phi_params[0];
    m_phi_max = phi_params[1];
    m_num_phi = static_cast<int>(std::llround(phi_params[2]));
    pp_fr.query("frequency_grid", m_omega_grid);
    utils::parser::queryWithParser(pp_fr, "particle_fraction", m_particle_fraction);
    std::string particle_filter_mode = "step";
    pp_fr.query("filter_mode", particle_filter_mode);
    m_particle_filter_is_latched = particle_filter_mode == "latched";
    std::string particle_filter_string;
    m_do_particle_filter = pp_fr.query(
        "filter_function(t,x,y,z,ux,uy,uz,w)", particle_filter_string);
    if (m_do_particle_filter) {
        utils::parser::Store_parserString(
            pp_fr,
            "filter_function(t,x,y,z,ux,uy,uz,w)",
            particle_filter_string);
        m_particle_filter_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(
                particle_filter_string, {"t", "x", "y", "z", "ux", "uy", "uz", "w"}));
        m_particle_filter_function = m_particle_filter_parser->compile<8>();
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_num_omega > 0 && m_num_theta > 0 && m_num_phi > 0,
        "Fourier radiation grid dimensions must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_omega_min > 0._rt && m_omega_max > 0._rt,
        rd_name + ".frequencies bounds must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_omega_grid == "linear" || m_omega_grid == "log",
        rd_name + ".frequency_grid must be either 'linear' or 'log'.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        particle_filter_mode == "step" || particle_filter_mode == "latched",
        rd_name + ".filter_mode must be either 'step' or 'latched'.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_particle_filter_is_latched || m_do_particle_filter,
        rd_name + ".filter_mode = latched requires a filter_function.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_particle_fraction > 0._rt && m_particle_fraction <= 1._rt,
        rd_name + ".particle_fraction must be in the interval (0, 1].");

    pp_fr.query("reset_after_output", m_reset_after_output);

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

    Vector<Real> h_grid_data(m_num_omega + 2*m_num_theta + 2*m_num_phi);
    Real* const h_omega = h_grid_data.data();
    Real* const h_sin_theta = h_omega + m_num_omega;
    Real* const h_cos_theta = h_sin_theta + m_num_theta;
    Real* const h_sin_phi = h_cos_theta + m_num_theta;
    Real* const h_cos_phi = h_sin_phi + m_num_phi;
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

    m_grid_data.resize(h_grid_data.size());
    Gpu::copy(Gpu::hostToDevice, h_grid_data.begin(), h_grid_data.end(), m_grid_data.begin());

    m_num_amp_streams = Gpu::numGpuStreams();
    m_amp.resize(
        static_cast<Long>(m_num_amp_streams) * fr::n_amp_components * m_num_grid_nodes, 0._rt);
}

FourierRadiationDeviceData
FourierRadiation::GetDeviceData ()
{
    Long stream_offset = 0;
#ifdef AMREX_USE_GPU
    stream_offset = static_cast<Long>(Gpu::Device::streamIndex())
        * fr::n_amp_components * m_num_grid_nodes;
#endif

    return FourierRadiationDeviceData{
        OmegaData(),
        SinThetaData(),
        CosThetaData(),
        SinPhiData(),
        CosPhiData(),
        m_amp.dataPtr() + stream_offset,
        m_num_omega,
        m_num_theta,
        m_num_phi,
        m_num_grid_nodes,
        m_particle_fraction};
}

void
FourierRadiation::GetIntensity (std::vector<Real>& intensity) const
{
    intensity.assign(m_num_grid_nodes, 0._rt);
    if (!m_enabled) {
        return;
    }

    Vector<Real> amp(
        static_cast<Long>(m_num_amp_streams) * fr::n_amp_components * m_num_grid_nodes);
    Gpu::copy(Gpu::deviceToHost, m_amp.begin(), m_amp.end(), amp.begin());
    ParallelDescriptor::ReduceRealSum(amp.data(), static_cast<int>(amp.size()));

    for (int i = 0; i < m_num_grid_nodes; ++i) {
        constexpr Real prefactor =
            1._rt / (16._rt * Math::pi<Real>() * Math::pi<Real>() * Math::pi<Real>()
                     * PhysConst::epsilon_0 * PhysConst::c);
        Real axr = 0._rt;
        Real axi = 0._rt;
        Real ayr = 0._rt;
        Real ayi = 0._rt;
        Real azr = 0._rt;
        Real azi = 0._rt;
        for (int istream = 0; istream < m_num_amp_streams; ++istream) {
            Real const* const stream_amp = amp.data()
                + static_cast<Long>(istream) * fr::n_amp_components * m_num_grid_nodes;
            axr += stream_amp[fr::amp_x_re * m_num_grid_nodes + i];
            axi += stream_amp[fr::amp_x_im * m_num_grid_nodes + i];
            ayr += stream_amp[fr::amp_y_re * m_num_grid_nodes + i];
            ayi += stream_amp[fr::amp_y_im * m_num_grid_nodes + i];
            azr += stream_amp[fr::amp_z_re * m_num_grid_nodes + i];
            azi += stream_amp[fr::amp_z_im * m_num_grid_nodes + i];
        }
        intensity[i] = prefactor * (axr*axr + axi*axi
                                  + ayr*ayr + ayi*ayi
                                  + azr*azr + azi*azi);
    }
}

void
FourierRadiation::Reset ()
{
    if (!m_enabled) {
        return;
    }

    Real* const amp = m_amp.dataPtr();
    int const n = static_cast<int>(m_amp.size());

    ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
    {
        amp[i] = 0._rt;
    });
}
