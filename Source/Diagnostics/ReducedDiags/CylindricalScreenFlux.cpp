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
#include <AMReX_GpuControl.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

CylindricalScreenFlux::CylindricalScreenFlux (const std::string& rd_name)
    : ReducedDiags{rd_name} {
#if !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "CylindricalScreenFlux is only available in RZ geometry.");
#else
    const amrex::ParmParse pp_rd_name(rd_name);
    utils::parser::getWithParser(pp_rd_name, "r0", m_r0);

    auto& warpx = WarpX::GetInstance();
    const amrex::Geometry& geom = warpx.Geom(0);
    m_z_min = geom.ProbLo(1);
    m_z_max = geom.ProbHi(1);
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
        m_t_max = std::numeric_limits<amrex::Real>::max();
        utils::parser::queryWithParser(pp_rd_name, "t_min", m_t_min);
        utils::parser::queryWithParser(pp_rd_name, "t_max", m_t_max);
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
            m_t_min <= m_t_max,
            "CylindricalScreenFlux.t_min must be <= t_max.");
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
        m_histogram.assign(histogram_size, 0.0);
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
    ofs << "[" << c++ << "]t_min(s)";
    ofs << m_sep;
    ofs << "[" << c++ << "]t_max(s)";
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
#if defined(WARPX_DIM_RZ)
    auto& warpx = WarpX::GetInstance();
    const auto& mypc = warpx.GetPartContainer();
    const bool do_output = m_intervals.contains(step + 1);
    const amrex::Real time = warpx.gett_new(0);
    const bool do_histogram =
        m_histogram_enabled && !m_histogram_written &&
        time >= m_t_min && time <= m_t_max;

    std::ofstream ofs;
    if (do_output && m_write_particles) {
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

        std::map<std::uint64_t, amrex::ParticleReal> current_r;

        const int nlevs = std::max(0, myspc.finestLevel() + 1);
        for (int lev = 0; lev < nlevs; ++lev) {
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti) {
                const auto& soa = pti.GetStructOfArrays();
                const auto& attribs = pti.GetAttribs();
                const auto* const AMREX_RESTRICT idcpu =
                    soa.GetIdCPUData().data();
                const auto* const AMREX_RESTRICT r = attribs[PIdx::r].dataPtr();
                const auto* const AMREX_RESTRICT z = attribs[PIdx::z].dataPtr();
                const auto* const AMREX_RESTRICT w = attribs[PIdx::w].dataPtr();
                const auto* const AMREX_RESTRICT ux =
                    attribs[PIdx::ux].dataPtr();
                const auto* const AMREX_RESTRICT uy =
                    attribs[PIdx::uy].dataPtr();
                const auto* const AMREX_RESTRICT uz =
                    attribs[PIdx::uz].dataPtr();
                const auto* const AMREX_RESTRICT theta =
                    attribs[PIdx::theta].dataPtr();

                const long np = pti.numParticles();
                for (long i = 0; i < np; ++i) {
                    const std::uint64_t packed_id = idcpu[i];
                    const amrex::ConstParticleIDWrapper pid{idcpu[i]};
                    if (!pid.is_valid()) {
                        continue;
                    }

                    const amrex::ParticleReal r_new = r[i];
                    current_r[packed_id] = r_new;

                    if (z[i] < m_z_min || z[i] > m_z_max) {
                        continue;
                    }

                    const auto prev =
                        species_state.m_previous_r.find(packed_id);
                    if (prev == species_state.m_previous_r.end()) {
                        continue;
                    }

                    const amrex::ParticleReal r_old = prev->second;
                    const amrex::ParticleReal ur =
                        ux[i] * std::cos(theta[i]) + uy[i] * std::sin(theta[i]);
                    if (r_old < m_r0 || r_new > m_r0 ||
                        ur >= amrex::ParticleReal(0.0)) {
                        continue;
                    }

                    const amrex::ParticleReal x = r_new * std::cos(theta[i]);
                    const amrex::ParticleReal y = r_new * std::sin(theta[i]);
                    const amrex::ParticleReal usq =
                        ux[i] * ux[i] + uy[i] * uy[i] + uz[i] * uz[i];
                    const amrex::ParticleReal gamma = std::sqrt(
                        amrex::ParticleReal(1.0) + usq * PhysConst::inv_c2);
                    const amrex::ParticleReal px = mass * ux[i];
                    const amrex::ParticleReal py = mass * uy[i];
                    const amrex::ParticleReal pz = mass * uz[i];
                    const amrex::ParticleReal ke_eV =
                        (gamma - amrex::ParticleReal(1.0)) * mass *
                        PhysConst::c * PhysConst::c * joule_to_eV;

                    if (do_output && m_write_particles) {
                        ofs << step + 1 << m_sep;
                        ofs << time << m_sep;
                        ofs << species_state.m_name << m_sep;
                        ofs << static_cast<amrex::Long>(pid) << m_sep;
                        ofs << x << m_sep;
                        ofs << y << m_sep;
                        ofs << z[i] << m_sep;
                        ofs << r_new << m_sep;
                        ofs << theta[i] << m_sep;
                        ofs << px << m_sep;
                        ofs << py << m_sep;
                        ofs << pz << m_sep;
                        ofs << ke_eV << m_sep;
                        ofs << ux[i] << m_sep;
                        ofs << uy[i] << m_sep;
                        ofs << uz[i] << m_sep;
                        ofs << w[i] << "\n";
                    }

                    if (do_histogram) {
                        constexpr amrex::Real pi =
                            3.141592653589793238462643383279502884;
                        const amrex::Real theta_norm =
                            (theta[i] + pi) / (2.0 * pi);
                        const amrex::Real z_norm =
                            (z[i] - m_z_min) / (m_z_max - m_z_min);
                        const amrex::Real energy_norm =
                            (ke_eV - m_energy_min) /
                            (m_energy_max - m_energy_min);
                        const int theta_bin =
                            static_cast<int>(std::floor(theta_norm * m_bins_theta));
                        const int z_bin =
                            static_cast<int>(std::floor(z_norm * m_bins_z));
                        const int energy_bin =
                            static_cast<int>(std::floor(energy_norm * m_bins_energy));

                        if (theta_bin >= 0 && theta_bin < m_bins_theta &&
                            z_bin >= 0 && z_bin < m_bins_z &&
                            energy_bin >= 0 && energy_bin < m_bins_energy) {
                            m_histogram[HistogramIndex(
                                species_counter, theta_bin, z_bin, energy_bin, 0)] +=
                                static_cast<amrex::Real>(w[i]);
                            m_histogram[HistogramIndex(
                                species_counter, theta_bin, z_bin, energy_bin, 1)] +=
                                1.0;
                        }
                    }
                }
            }
        }

        species_state.m_previous_r = std::move(current_r);
    }

    if (m_histogram_enabled && !m_histogram_written && time >= m_t_max) {
        amrex::ParallelDescriptor::ReduceRealSum(
            m_histogram.data(), static_cast<int>(m_histogram.size()),
            amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor()) {
            WriteHistogram(step, time, m_histogram);
        }
        m_histogram_written = true;
    }
#else
    static_cast<void>(step);
#endif
}

void
CylindricalScreenFlux::WriteToFile (int /*step*/) const {}

void
CylindricalScreenFlux::WriteHistogram (
    int step, amrex::Real time, std::vector<amrex::Real> const& histogram) const
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
                    ofs << m_t_min << m_sep;
                    ofs << m_t_max << m_sep;
                    ofs << sum_weight << m_sep;
                    ofs << count << "\n";
                }
            }
        }
    }
}
