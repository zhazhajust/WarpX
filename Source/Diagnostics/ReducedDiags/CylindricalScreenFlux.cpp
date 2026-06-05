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

    std::string restart_chkfile;
    const amrex::ParmParse pp_amr("amr");
    pp_amr.query("restart", restart_chkfile);
    const bool is_not_restart = restart_chkfile.empty();
    m_write_header = is_not_restart || !amrex::FileExists(m_file_name);

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

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
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
#endif
}

void
CylindricalScreenFlux::ComputeDiags (int step) {
#if defined(WARPX_DIM_RZ)
    auto& warpx = WarpX::GetInstance();
    const auto& mypc = warpx.GetPartContainer();
    const bool do_output = m_intervals.contains(step + 1);
    const amrex::Real time = warpx.gett_new(0);

    std::ofstream ofs;
    if (do_output) {
        ofs.open(m_file_name, std::ofstream::out | std::ofstream::app);
        ofs << std::fixed << std::setprecision(m_precision) << std::scientific;
    }

    for (auto& species_state : m_species) {
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

                    if (!do_output) {
                        continue;
                    }
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
            }
        }

        species_state.m_previous_r = std::move(current_r);
    }
#else
    static_cast<void>(step);
#endif
}

void
CylindricalScreenFlux::WriteToFile (int /*step*/) const {}
