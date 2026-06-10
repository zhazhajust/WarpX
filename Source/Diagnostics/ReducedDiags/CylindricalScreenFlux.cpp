/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "CylindricalScreenFlux.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
    struct ScreenEvent {
        std::uint64_t m_packed_id = 0;
        amrex::ParticleReal m_x = 0.0;
        amrex::ParticleReal m_y = 0.0;
        amrex::ParticleReal m_z = 0.0;
        amrex::ParticleReal m_r = 0.0;
        amrex::ParticleReal m_theta = 0.0;
        amrex::ParticleReal m_px = 0.0;
        amrex::ParticleReal m_py = 0.0;
        amrex::ParticleReal m_pz = 0.0;
        amrex::ParticleReal m_ke_eV = 0.0;
        amrex::ParticleReal m_ux = 0.0;
        amrex::ParticleReal m_uy = 0.0;
        amrex::ParticleReal m_uz = 0.0;
        amrex::ParticleReal m_w = 0.0;
    };

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal NormalizeTheta (amrex::ParticleReal theta)
    {
        constexpr amrex::ParticleReal pi =
            amrex::ParticleReal(3.141592653589793238462643383279502884);
        constexpr amrex::ParticleReal two_pi = amrex::ParticleReal(2.0) * pi;
        while (theta < -pi) {
            theta += two_pi;
        }
        while (theta >= pi) {
            theta -= two_pi;
        }
        return theta;
    }
#endif
}

CylindricalScreenFlux::CylindricalScreenFlux (const std::string& rd_name)
    : ReducedDiags{rd_name} {
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_3D)
    WARPX_ABORT_WITH_MESSAGE(
        "CylindricalScreenFlux is only available in 2D XZ, 3D, and RZ geometry.");
#else
    const amrex::ParmParse pp_rd_name(rd_name);
    utils::parser::getWithParser(pp_rd_name, "r0", m_r0);
    utils::parser::queryWithParser(pp_rd_name, "center_x", m_center_x);
#if defined(WARPX_DIM_3D)
    utils::parser::queryWithParser(pp_rd_name, "center_y", m_center_y);
#endif
#if defined(WARPX_DIM_XZ)
    utils::parser::queryWithParser(pp_rd_name, "theta", m_theta_2d);
    pp_rd_name.query("two_sided", m_two_sided_2d);
    m_theta_2d = NormalizeTheta(m_theta_2d);
#endif

    auto& warpx = WarpX::GetInstance();
    const amrex::Geometry& geom = warpx.Geom(0);
#if defined(WARPX_DIM_3D)
    m_z_min = geom.ProbLo(2);
    m_z_max = geom.ProbHi(2);
#else
    m_z_min = geom.ProbLo(1);
    m_z_max = geom.ProbHi(1);
#endif
    utils::parser::queryWithParser(pp_rd_name, "z_min", m_z_min);
    utils::parser::queryWithParser(pp_rd_name, "z_max", m_z_max);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_r0 >= 0.0,
                                     "CylindricalScreenFlux.r0 must be >= 0.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_min <= m_z_max, "CylindricalScreenFlux.z_min must be <= z_max.");

    m_extension = "csv";
    m_file_name = m_path + m_rd_name + "." + m_extension;
    pp_rd_name.query("file_name", m_file_name);
    pp_rd_name.query("write_particles", m_write_particles);

    m_histogram_enabled =
        pp_rd_name.query("bins_theta", m_bins_theta) ||
        pp_rd_name.query("bins_z", m_bins_z) ||
        pp_rd_name.query("bins_energy", m_bins_energy) ||
        pp_rd_name.query("histogram_file_name", m_histogram_file_name);
    if (m_histogram_enabled) {
        utils::parser::getWithParser(pp_rd_name, "bins_theta", m_bins_theta);
        utils::parser::getWithParser(pp_rd_name, "bins_z", m_bins_z);
        utils::parser::getWithParser(pp_rd_name, "bins_energy", m_bins_energy);
        utils::parser::getWithParser(pp_rd_name, "energy_min", m_energy_min);
        utils::parser::getWithParser(pp_rd_name, "energy_max", m_energy_max);
        const bool has_time_interval =
            utils::parser::queryWithParser(
                pp_rd_name, "time_interval", m_time_interval) ||
            utils::parser::queryWithParser(
                pp_rd_name, "time_intervel", m_time_interval);
        m_histogram_file_name = m_path + m_rd_name + "_histogram." + m_extension;
        pp_rd_name.query("histogram_file_name", m_histogram_file_name);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_bins_theta > 0, "CylindricalScreenFlux.bins_theta must be > 0.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_bins_z > 0, "CylindricalScreenFlux.bins_z must be > 0.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_bins_energy > 0, "CylindricalScreenFlux.bins_energy must be > 0.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_energy_min < m_energy_max,
            "CylindricalScreenFlux.energy_min must be < energy_max.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            has_time_interval,
            "CylindricalScreenFlux.time_interval must be specified.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_time_interval > 0.0,
            "CylindricalScreenFlux.time_interval must be > 0.");
        m_interval_stop = m_time_interval;
    }

    std::string restart_chkfile;
    const amrex::ParmParse pp_amr("amr");
    pp_amr.query("restart", restart_chkfile);
    const bool is_not_restart = restart_chkfile.empty();
    m_write_header = is_not_restart || !amrex::FileExists(m_file_name);
    m_write_histogram_header =
        m_histogram_enabled &&
        (is_not_restart || !amrex::FileExists(m_histogram_file_name));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_write_particles || m_histogram_enabled,
        "CylindricalScreenFlux.write_particles=0 requires histogram output "
        "parameters.");

    const auto& mypc = warpx.GetPartContainer();
    const auto species_names = mypc.GetSpeciesNames();

    std::vector<std::string> selected_species_names;
    const bool has_selected_species =
        pp_rd_name.queryarr("species", selected_species_names);
    std::set<std::string> selected_species;
    if (has_selected_species) {
        selected_species.insert(selected_species_names.begin(),
                                selected_species_names.end());
    }

    for (int i = 0; i < mypc.nSpecies(); ++i) {
        const auto& species = mypc.GetParticleContainer(i);
        const bool is_selected = !has_selected_species ||
                                 selected_species.count(species_names[i]) != 0;
        if (!is_selected) {
            continue;
        }
        if (!has_selected_species && species.AmIA<PhysicalSpecies::photon>()) {
            continue;
        }

        SpeciesState state;
        state.m_index = i;
        state.m_name = species_names[i];
        state.m_previous_radius_name =
            "__" + m_rd_name + "_" + species_names[i] + "_previous_radius";
        const auto real_names = species.GetRealSoANames();
        const bool has_previous_radius =
            std::find(
                real_names.begin(), real_names.end(),
                state.m_previous_radius_name) != real_names.end();
        if (!has_previous_radius) {
            const int communicate = 1;
            mypc.GetParticleContainer(i).AddRealComp(
                state.m_previous_radius_name, communicate);
        }
        m_species.push_back(std::move(state));
    }

    if (has_selected_species) {
        for (const auto& species_name : selected_species) {
            const auto found = std::find(species_names.begin(),
                                         species_names.end(), species_name);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                found != species_names.end(),
                "Unknown species for CylindricalScreenFlux reduced "
                "diagnostic: " +
                    species_name);
        }
    }

    if (m_histogram_enabled) {
        constexpr int ncomp = 2;
        const auto histogram_size =
            static_cast<std::size_t>(m_species.size()) *
            static_cast<std::size_t>(m_bins_theta) *
            static_cast<std::size_t>(m_bins_z) *
            static_cast<std::size_t>(m_bins_energy) * ncomp;
        m_histogram.resize(histogram_size);
        m_histogram_device.resize(histogram_size);
        ResetHistogram();
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (m_write_particles && m_write_header) {
            WriteParticleHeader();
        }
        if (m_write_histogram_header) {
            WriteHistogramHeader();
        }
    }
#endif
}

void
CylindricalScreenFlux::WriteParticleHeader () const {
    std::ofstream ofs{m_file_name, std::ofstream::out};
    int c = 0;
    ofs << "#";
    ofs << "[" << c++ << "]step()";
    ofs << m_sep;
    ofs << "[" << c++ << "]time(s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]species()";
    ofs << m_sep;
    ofs << "[" << c++ << "]id()";
    ofs << m_sep;
    ofs << "[" << c++ << "]x(m)";
    ofs << m_sep;
    ofs << "[" << c++ << "]y(m)";
    ofs << m_sep;
    ofs << "[" << c++ << "]z(m)";
    ofs << m_sep;
    ofs << "[" << c++ << "]r(m)";
    ofs << m_sep;
    ofs << "[" << c++ << "]theta(rad)";
    ofs << m_sep;
    ofs << "[" << c++ << "]px(kg*m/s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]py(kg*m/s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]pz(kg*m/s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]KE_eV(eV)";
    ofs << m_sep;
    ofs << "[" << c++ << "]ux(m/s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]uy(m/s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]uz(m/s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]weight()";
    ofs << "\n";
}

void
CylindricalScreenFlux::WriteHistogramHeader () const {
    std::ofstream ofs{m_histogram_file_name, std::ofstream::out};
    int c = 0;
    ofs << "#";
    ofs << "[" << c++ << "]step()";
    ofs << m_sep;
    ofs << "[" << c++ << "]time(s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]interval_index()";
    ofs << m_sep;
    ofs << "[" << c++ << "]interval_start(s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]interval_stop(s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]species()";
    ofs << m_sep;
    ofs << "[" << c++ << "]theta_bin()";
    ofs << m_sep;
    ofs << "[" << c++ << "]z_bin()";
    ofs << m_sep;
    ofs << "[" << c++ << "]energy_bin()";
    ofs << m_sep;
    ofs << "[" << c++ << "]theta_min(rad)";
    ofs << m_sep;
    ofs << "[" << c++ << "]theta_max(rad)";
    ofs << m_sep;
    ofs << "[" << c++ << "]z_min(m)";
    ofs << m_sep;
    ofs << "[" << c++ << "]z_max(m)";
    ofs << m_sep;
    ofs << "[" << c++ << "]energy_min(eV)";
    ofs << m_sep;
    ofs << "[" << c++ << "]energy_max(eV)";
    ofs << m_sep;
    ofs << "[" << c++ << "]sum_weight()";
    ofs << m_sep;
    ofs << "[" << c++ << "]count()";
    ofs << "\n";
}

std::size_t
CylindricalScreenFlux::HistogramIndex (
    int species_index, int theta_bin, int z_bin, int energy_bin, int component) const
{
    constexpr int ncomp = 2;
    return static_cast<std::size_t>(
        (((species_index * m_bins_theta + theta_bin) * m_bins_z + z_bin) *
             m_bins_energy +
         energy_bin) *
            ncomp +
        component);
}

void
CylindricalScreenFlux::ComputeDiags (int step) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
    auto& warpx = WarpX::GetInstance();
    const auto& mypc = warpx.GetPartContainer();
    const bool do_output = m_intervals.contains(step + 1);
    const bool can_detect_crossing = m_has_previous_radius;
    const bool write_events = can_detect_crossing && do_output && m_write_particles;
    const amrex::Real time = warpx.gett_new(0);

    while (m_histogram_enabled && time > m_interval_stop) {
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, m_histogram_device.begin(),
            m_histogram_device.end(), m_histogram.begin());
        amrex::ParallelDescriptor::ReduceRealSum(
            m_histogram.data(), static_cast<int>(m_histogram.size()),
            amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor()) {
            WriteHistogram(
                step, m_interval_stop, m_histogram_interval_index,
                m_interval_start, m_interval_stop, m_histogram);
        }
        ResetHistogram();
        ++m_histogram_interval_index;
        m_interval_start = m_interval_stop;
        m_interval_stop += m_time_interval;
    }

    std::ofstream ofs;
    if (write_events) {
        ofs.open(m_file_name, std::ofstream::out | std::ofstream::app);
        ofs << std::fixed << std::setprecision(m_precision) << std::scientific;
    }

    for (int species_counter = 0; species_counter < static_cast<int>(m_species.size());
         ++species_counter) {
        auto& species_state = m_species[species_counter];
        auto& myspc = mypc.GetParticleContainer(species_state.m_index);
        const amrex::ParticleReal mass = myspc.getMass();
        const amrex::ParticleReal joule_to_eV =
            amrex::ParticleReal(1.0) / PhysConst::q_e;

        const int nlevs = std::max(0, myspc.finestLevel() + 1);
        for (int lev = 0; lev < nlevs; ++lev) {
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti) {
                const auto& soa = pti.GetStructOfArrays();
                const auto& attribs = pti.GetAttribs();
                const long np = pti.numParticles();
                if (np == 0) {
                    continue;
                }

                const auto* const AMREX_RESTRICT idcpu =
                    soa.GetIdCPUData().data();
                const auto* const AMREX_RESTRICT w =
                    attribs[PIdx::w].dataPtr();
                const auto* const AMREX_RESTRICT ux =
                    attribs[PIdx::ux].dataPtr();
                const auto* const AMREX_RESTRICT uy =
                    attribs[PIdx::uy].dataPtr();
                const auto* const AMREX_RESTRICT uz =
                    attribs[PIdx::uz].dataPtr();

                auto* const AMREX_RESTRICT previous_radius =
                    pti.GetAttribs(species_state.m_previous_radius_name).dataPtr();
#if defined(WARPX_DIM_RZ)
                const auto* const AMREX_RESTRICT r = attribs[PIdx::r].dataPtr();
                const auto* const AMREX_RESTRICT z = attribs[PIdx::z].dataPtr();
                const auto* const AMREX_RESTRICT theta =
                    attribs[PIdx::theta].dataPtr();
#elif defined(WARPX_DIM_XZ)
                const auto* const AMREX_RESTRICT x = attribs[PIdx::x].dataPtr();
                const auto* const AMREX_RESTRICT z = attribs[PIdx::z].dataPtr();
#elif defined(WARPX_DIM_3D)
                const auto* const AMREX_RESTRICT x = attribs[PIdx::x].dataPtr();
                const auto* const AMREX_RESTRICT y = attribs[PIdx::y].dataPtr();
                const auto* const AMREX_RESTRICT z = attribs[PIdx::z].dataPtr();
#endif

                amrex::Gpu::DeviceVector<ScreenEvent> events;
                amrex::Gpu::DeviceVector<int> event_count;
                ScreenEvent* event_data = nullptr;
                int* event_count_data = nullptr;
                if (write_events) {
                    events.resize(np);
                    event_count.resize(1, 0);
                    event_data = events.dataPtr();
                    event_count_data = event_count.dataPtr();
                }

                amrex::Real* const histogram =
                    m_histogram_enabled ? m_histogram_device.dataPtr() : nullptr;
                const bool histogram_enabled = m_histogram_enabled;
                const bool histogram_active =
                    can_detect_crossing && m_histogram_enabled &&
                    time <= m_interval_stop;

                const auto r0 = m_r0;
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
                const auto center_x = m_center_x;
#endif
#if defined(WARPX_DIM_3D)
                const auto center_y = m_center_y;
#endif
                const auto z_min = m_z_min;
                const auto z_max = m_z_max;
                const auto energy_min = m_energy_min;
                const auto energy_max = m_energy_max;
                const int bins_theta = m_bins_theta;
                const int bins_z = m_bins_z;
                const int bins_energy = m_bins_energy;
#if defined(WARPX_DIM_XZ)
                const bool two_sided_2d = m_two_sided_2d;
                const auto theta_2d = m_theta_2d;
#endif
                const bool detect_crossing = can_detect_crossing;
                constexpr int ncomp = 2;

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE (long i)
                    {
                        const amrex::ConstParticleIDWrapper pid{idcpu[i]};
                        if (!pid.is_valid()) {
                            return;
                        }

                        amrex::ParticleReal x_event = 0.0;
                        amrex::ParticleReal y_event = 0.0;
                        amrex::ParticleReal z_event = 0.0;
                        amrex::ParticleReal r_new = 0.0;
                        amrex::ParticleReal theta_event = 0.0;
                        bool crossed = false;

#if defined(WARPX_DIM_RZ)
                        r_new = r[i];
                        theta_event = NormalizeTheta(theta[i]);
                        z_event = z[i];
                        x_event = r_new * std::cos(theta_event);
                        y_event = r_new * std::sin(theta_event);
                        const amrex::ParticleReal ur =
                            ux[i] * std::cos(theta_event) +
                            uy[i] * std::sin(theta_event);
                        const amrex::ParticleReal r_old = previous_radius[i];
                        previous_radius[i] = r_new;
                        crossed = detect_crossing &&
                                  r_old >= r0 && r_new <= r0 &&
                                  ur < amrex::ParticleReal(0.0);
#elif defined(WARPX_DIM_XZ)
                        const amrex::ParticleReal signed_r_new = x[i] - center_x;
                        const amrex::ParticleReal signed_r_old = previous_radius[i];
                        previous_radius[i] = signed_r_new;
                        z_event = z[i];
                        x_event = x[i];
                        y_event = 0.0;
                        r_new = amrex::Math::abs(signed_r_new);

                        const bool crossed_right =
                            detect_crossing && signed_r_old >= r0 &&
                            signed_r_new <= r0 &&
                            ux[i] < amrex::ParticleReal(0.0);
                        const bool crossed_left =
                            detect_crossing && two_sided_2d &&
                            signed_r_old <= -r0 &&
                            signed_r_new >= -r0 && ux[i] > amrex::ParticleReal(0.0);
                        crossed = crossed_right || crossed_left;
                        theta_event = crossed_left
                                          ? NormalizeTheta(
                                                theta_2d +
                                                amrex::ParticleReal(
                                                    3.141592653589793238462643383279502884))
                                          : theta_2d;
#elif defined(WARPX_DIM_3D)
                        const amrex::ParticleReal dx = x[i] - center_x;
                        const amrex::ParticleReal dy = y[i] - center_y;
                        r_new = std::sqrt(dx * dx + dy * dy);
                        const amrex::ParticleReal inv_r =
                            r_new > amrex::ParticleReal(0.0)
                                ? amrex::ParticleReal(1.0) / r_new
                                : amrex::ParticleReal(0.0);
                        const amrex::ParticleReal costheta =
                            r_new > amrex::ParticleReal(0.0)
                                ? dx * inv_r
                                : amrex::ParticleReal(1.0);
                        const amrex::ParticleReal sintheta =
                            r_new > amrex::ParticleReal(0.0)
                                ? dy * inv_r
                                : amrex::ParticleReal(0.0);
                        theta_event = std::atan2(dy, dx);
                        z_event = z[i];
                        x_event = x[i];
                        y_event = y[i];
                        const amrex::ParticleReal ur =
                            ux[i] * costheta + uy[i] * sintheta;
                        const amrex::ParticleReal r_old = previous_radius[i];
                        previous_radius[i] = r_new;
                        crossed = detect_crossing &&
                                  r_old >= r0 && r_new <= r0 &&
                                  ur < amrex::ParticleReal(0.0);
#endif

                        if (!crossed || z_event < z_min || z_event > z_max) {
                            return;
                        }

                        const amrex::ParticleReal usq =
                            ux[i] * ux[i] + uy[i] * uy[i] + uz[i] * uz[i];
                        const amrex::ParticleReal gamma = std::sqrt(
                            amrex::ParticleReal(1.0) + usq * PhysConst::inv_c2);
                        const amrex::ParticleReal ke_eV =
                            (gamma - amrex::ParticleReal(1.0)) * mass *
                            PhysConst::c * PhysConst::c * joule_to_eV;

                        if (histogram_active) {
                            constexpr amrex::Real pi =
                                3.141592653589793238462643383279502884;
                            const amrex::Real theta_norm =
                                (static_cast<amrex::Real>(theta_event) + pi) /
                                (2.0 * pi);
                            const amrex::Real z_norm =
                                (static_cast<amrex::Real>(z_event) - z_min) /
                                (z_max - z_min);
                            const amrex::Real energy_norm =
                                (static_cast<amrex::Real>(ke_eV) - energy_min) /
                                (energy_max - energy_min);
                            const int theta_bin = static_cast<int>(
                                amrex::Math::floor(theta_norm * bins_theta));
                            const int z_bin = static_cast<int>(
                                amrex::Math::floor(z_norm * bins_z));
                            const int energy_bin = static_cast<int>(
                                amrex::Math::floor(energy_norm * bins_energy));

                            if (theta_bin >= 0 && theta_bin < bins_theta &&
                                z_bin >= 0 && z_bin < bins_z &&
                                energy_bin >= 0 && energy_bin < bins_energy) {
                                const std::size_t histogram_index =
                                    static_cast<std::size_t>(
                                        (((species_counter * bins_theta + theta_bin) *
                                              bins_z +
                                          z_bin) *
                                             bins_energy +
                                         energy_bin) *
                                            ncomp);
                                amrex::Gpu::Atomic::AddNoRet(
                                    &histogram[histogram_index],
                                    static_cast<amrex::Real>(w[i]));
                                amrex::Gpu::Atomic::AddNoRet(
                                    &histogram[histogram_index + 1],
                                    amrex::Real(1.0));
                            }
                        } else if (histogram_enabled) {
                            amrex::ignore_unused(histogram);
                        }

                        if (event_data != nullptr) {
                            const int event_index =
                                amrex::Gpu::Atomic::Add(event_count_data, 1);
                            event_data[event_index] = ScreenEvent{
                                idcpu[i],
                                x_event,
                                y_event,
                                z_event,
                                r_new,
                                theta_event,
                                mass * ux[i],
                                mass * uy[i],
                                mass * uz[i],
                                ke_eV,
                                ux[i],
                                uy[i],
                                uz[i],
                                w[i]};
                        }
                    });

                if (write_events) {
                    amrex::Gpu::HostVector<int> h_event_count(1);
                    amrex::Gpu::copy(
                        amrex::Gpu::deviceToHost, event_count.begin(),
                        event_count.end(), h_event_count.begin());
                    const int num_events = h_event_count[0];
                    if (num_events > 0) {
                        amrex::Gpu::HostVector<ScreenEvent> h_events(num_events);
                        amrex::Gpu::copy(
                            amrex::Gpu::deviceToHost, events.begin(),
                            events.begin() + num_events, h_events.begin());
                        for (const auto& event : h_events) {
                            const amrex::ConstParticleIDWrapper pid{
                                event.m_packed_id};
                            ofs << step + 1 << m_sep;
                            ofs << time << m_sep;
                            ofs << species_state.m_name << m_sep;
                            ofs << static_cast<amrex::Long>(pid) << m_sep;
                            ofs << event.m_x << m_sep;
                            ofs << event.m_y << m_sep;
                            ofs << event.m_z << m_sep;
                            ofs << event.m_r << m_sep;
                            ofs << event.m_theta << m_sep;
                            ofs << event.m_px << m_sep;
                            ofs << event.m_py << m_sep;
                            ofs << event.m_pz << m_sep;
                            ofs << event.m_ke_eV << m_sep;
                            ofs << event.m_ux << m_sep;
                            ofs << event.m_uy << m_sep;
                            ofs << event.m_uz << m_sep;
                            ofs << event.m_w << "\n";
                        }
                    }
                }
            }
        }
    }
    m_has_previous_radius = true;

    if (m_histogram_enabled && time >= m_interval_stop) {
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, m_histogram_device.begin(),
            m_histogram_device.end(), m_histogram.begin());
        amrex::ParallelDescriptor::ReduceRealSum(
            m_histogram.data(), static_cast<int>(m_histogram.size()),
            amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor()) {
            WriteHistogram(
                step, m_interval_stop, m_histogram_interval_index,
                m_interval_start, m_interval_stop, m_histogram);
        }
        ResetHistogram();
        ++m_histogram_interval_index;
        m_interval_start = m_interval_stop;
        m_interval_stop += m_time_interval;
    }
#else
    static_cast<void>(step);
#endif
}

void
CylindricalScreenFlux::WriteToFile (int /*step*/) const {}

void
CylindricalScreenFlux::ResetHistogram ()
{
    std::fill(m_histogram.begin(), m_histogram.end(), 0.0);
    if (!m_histogram_device.empty()) {
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, m_histogram.begin(), m_histogram.end(),
            m_histogram_device.begin());
    }
}

void
CylindricalScreenFlux::WriteHistogram (
    int step,
    amrex::Real time,
    int interval_index,
    amrex::Real interval_start,
    amrex::Real interval_stop,
    std::vector<amrex::Real> const& histogram) const
{
    std::ofstream ofs{m_histogram_file_name, std::ofstream::out | std::ofstream::app};
    ofs << std::fixed << std::setprecision(m_precision) << std::scientific;

    constexpr amrex::Real pi = 3.141592653589793238462643383279502884;
    const amrex::Real dtheta = 2.0 * pi / static_cast<amrex::Real>(m_bins_theta);
    const amrex::Real dz = (m_z_max - m_z_min) / static_cast<amrex::Real>(m_bins_z);
    const amrex::Real denergy =
        (m_energy_max - m_energy_min) / static_cast<amrex::Real>(m_bins_energy);

    for (int ispecies = 0; ispecies < static_cast<int>(m_species.size()); ++ispecies) {
        for (int itheta = 0; itheta < m_bins_theta; ++itheta) {
            const amrex::Real theta_min = -pi + itheta * dtheta;
            const amrex::Real theta_max = theta_min + dtheta;
            for (int iz = 0; iz < m_bins_z; ++iz) {
                const amrex::Real z_min = m_z_min + iz * dz;
                const amrex::Real z_max = z_min + dz;
                for (int ie = 0; ie < m_bins_energy; ++ie) {
                    const auto sum_weight = histogram[HistogramIndex(
                        ispecies, itheta, iz, ie, 0)];
                    const auto count = histogram[HistogramIndex(
                        ispecies, itheta, iz, ie, 1)];
                    if (sum_weight == 0.0 && count == 0.0) {
                        continue;
                    }

                    const amrex::Real energy_min = m_energy_min + ie * denergy;
                    const amrex::Real energy_max = energy_min + denergy;
                    ofs << step + 1 << m_sep;
                    ofs << time << m_sep;
                    ofs << interval_index << m_sep;
                    ofs << interval_start << m_sep;
                    ofs << interval_stop << m_sep;
                    ofs << m_species[ispecies].m_name << m_sep;
                    ofs << itheta << m_sep;
                    ofs << iz << m_sep;
                    ofs << ie << m_sep;
                    ofs << theta_min << m_sep;
                    ofs << theta_max << m_sep;
                    ofs << z_min << m_sep;
                    ofs << z_max << m_sep;
                    ofs << energy_min << m_sep;
                    ofs << energy_max << m_sep;
                    ofs << sum_weight << m_sep;
                    ofs << count << "\n";
                }
            }
        }
    }
}
