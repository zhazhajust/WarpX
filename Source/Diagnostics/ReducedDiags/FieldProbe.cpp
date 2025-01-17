/* Copyright 2021 Lorenzo Giacomel, Elisa Rheaume, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldProbe.H"
#include "FieldProbeParticleContainer.H"
#include "FieldSolver/Fields.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePosition.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array.H>
#include <AMReX_Config.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ParIter.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_Reduce.H>
#include <AMReX_Geometry.H>
#include <AMReX_StructOfArrays.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace amrex;
using namespace warpx::fields;

// constructor

FieldProbe::FieldProbe (const std::string& rd_name)
: ReducedDiags{rd_name}, m_probe(&WarpX::GetInstance())
{

    // read number of levels
    int nLevel = 0;
    const amrex::ParmParse pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    nLevel += 1;

    /* Obtain input data from parsing inputs file.
     * For the case of a single particle:
     *     Define x, y, and z of particle
     *     Define whether or not to integrate fields
     * For the case of a line detector:
     *     Define x, y, and z of end of line point 1
     *     Define x, y, and z of end of line point 2
     *     Define resolution to determine number of particles
     *     Define whether ot not to integrate fields
     * For the case of a plane detector:
     *     Define a vector normal to the detector plane
     *     Define a vector in the "up" direction of the plane
     *     Define the size of the plane (width of half square)
     *     Define resolution to determine number of particles
     *     Define whether ot not to integrate fields
     */
    const amrex::ParmParse pp_rd_name(rd_name);
    std::string m_probe_geometry_str = "Point";
    pp_rd_name.query("probe_geometry", m_probe_geometry_str);

    if (m_probe_geometry_str == "Point")
    {
        m_probe_geometry = DetectorGeometry::Point;
#if !defined(WARPX_DIM_1D_Z)
        utils::parser::getWithParser(
            pp_rd_name, "x_probe", x_probe);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        utils::parser::getWithParser(
            pp_rd_name, "y_probe", y_probe);
#endif
        utils::parser::getWithParser(
            pp_rd_name, "z_probe", z_probe);
    }
    else if (m_probe_geometry_str == "Line")
    {
        m_probe_geometry = DetectorGeometry::Line;
#if !defined(WARPX_DIM_1D_Z)
        utils::parser::queryWithParser(pp_rd_name, "x_probe", x_probe);
        utils::parser::queryWithParser(pp_rd_name, "x1_probe", x1_probe);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        utils::parser::queryWithParser(pp_rd_name, "y_probe", y_probe);
        utils::parser::queryWithParser(pp_rd_name, "y1_probe", y1_probe);
#endif
        utils::parser::getWithParser(pp_rd_name, "z_probe", z_probe);
        utils::parser::getWithParser(pp_rd_name, "z1_probe", z1_probe);
        utils::parser::getWithParser(pp_rd_name, "resolution", m_resolution);
    }
    else if (m_probe_geometry_str == "Plane")
    {
#if defined(WARPX_DIM_1D_Z)
        WARPX_ABORT_WITH_MESSAGE(
            "Plane probe should be used in a 2D or 3D simulation only");
#endif
        m_probe_geometry = DetectorGeometry::Plane;
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        utils::parser::queryWithParser(pp_rd_name, "y_probe", y_probe);
        utils::parser::queryWithParser(pp_rd_name, "target_normal_x", target_normal_x);
        utils::parser::queryWithParser(pp_rd_name, "target_normal_y", target_normal_y);
        utils::parser::queryWithParser(pp_rd_name, "target_normal_z", target_normal_z);
        utils::parser::queryWithParser(pp_rd_name, "target_up_y", target_up_y);
#endif
        utils::parser::queryWithParser(pp_rd_name, "x_probe", x_probe);
        utils::parser::getWithParser(pp_rd_name, "z_probe", z_probe);
        utils::parser::queryWithParser(pp_rd_name, "target_up_x", target_up_x);
        utils::parser::queryWithParser(pp_rd_name, "target_up_z", target_up_z);
        utils::parser::queryWithParser(pp_rd_name, "detector_radius", detector_radius);
        utils::parser::getWithParser(pp_rd_name, "resolution", m_resolution);
    }
    else
    {
        WARPX_ABORT_WITH_MESSAGE(
            "Invalid probe geometry '" + m_probe_geometry_str
            + "'. Valid geometries are Point, Line or Plane."
        );
    }
    pp_rd_name.query("integrate", m_field_probe_integrate);
    utils::parser::queryWithParser(pp_rd_name, "interp_order", interp_order);
    pp_rd_name.query("do_moving_window_FP", do_moving_window_FP);

    utils::parser::queryWithParser(pp_rd_name, "start_step", start_step);
    utils::parser::queryWithParser(pp_rd_name, "stop_step", stop_step);
    utils::parser::queryWithParser(pp_rd_name, "stop_move_step", stop_move_step);
    // utils::parser::queryWithParser(pp_rd_name, "max_level", max_level);
    // if(nLevel > max_level + 1) nLevel = max_level + 1;

    utils::parser::queryWithParser(pp_rd_name, "start_time", start_time);

    bool raw_fields;
    const bool raw_fields_specified = pp_rd_name.query("raw_fields", raw_fields);
    if (raw_fields_specified) {
        WARPX_ABORT_WITH_MESSAGE("The field probe raw_fields options is obsolete. To get the equivalent, set interp_order = 0");
    }

    if (WarpX::gamma_boost > 1.0_rt)
    {
        ablastr::warn_manager::WMRecordWarning(
            "Boosted Frame Invalid",
            "The FieldProbe Diagnostic will not record lab-frame, but boosted frame data.",
            ablastr::warn_manager::WarnPriority::low);
    }

    // ensure assumption holds: we read the fields in the interpolation kernel as they are,
    // without further communication of guard/ghost/halo regions
    int particle_shape;
    const ParmParse pp_algo("algo");
    utils::parser::getWithParser(pp_algo, "particle_shape", particle_shape);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(interp_order <= particle_shape ,
                                     "Field probe interp_order should be less than or equal to algo.particle_shape");
    if (ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // // open file
            // std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};

            // // write header row
            // int c = 0;
            // ofs << "[" << c++ << "]step()";
            // ofs << m_sep;
            // ofs << "[" << c++ << "]time(s)";
            // maps FieldProbe observables to units

            // IO process
            std::unordered_map< int, std::string > u_map;

            if (m_field_probe_integrate)
            {
                u_map =
                {
                    {FieldProbePIdx::Ex , "-(V*s/m)"},
                    {FieldProbePIdx::Ey , "-(V*s/m)"},
                    {FieldProbePIdx::Ez , "-(V*s/m)"},
                    {FieldProbePIdx::Bx , "-(T*s)"},
                    {FieldProbePIdx::By , "-(T*s)"},
                    {FieldProbePIdx::Bz , "-(T*s)"},
                    {FieldProbePIdx::S , "-(W*s/m^2)"}
                };
            }
            else
            {
                u_map =
                {
                    {FieldProbePIdx::Ex , "-(V/m)"},
                    {FieldProbePIdx::Ey , "-(V/m)"},
                    {FieldProbePIdx::Ez , "-(V/m)"},
                    {FieldProbePIdx::Bx , "-(T)"},
                    {FieldProbePIdx::By , "-(T)"},
                    {FieldProbePIdx::Bz , "-(T)"},
                    {FieldProbePIdx::S , "-(W/m^2)"}
                };
            }
            // for (int lev = 0; lev < nLevel; ++lev)
            for (int lev = 0; lev < 1; ++lev)
            {
                auto filename = lev > 0 ? m_path + m_rd_name + "_lvl_" + std::to_string(lev) + "." + m_extension:
                    m_path + m_rd_name + "." + m_extension;
                std::ofstream ofs{filename, std::ofstream::out};

                // write header row
                int c = 0;
                ofs << "[" << c++ << "]step()";
                ofs << m_sep;
                ofs << "[" << c++ << "]time(s)";

                ofs << m_sep;
                ofs << "[" << c++ << "]part_x_lev" + std::to_string(lev) + "-(m)";
                ofs << m_sep;
                ofs << "[" << c++ << "]part_y_lev" + std::to_string(lev) + "-(m)";
                ofs << m_sep;
                ofs << "[" << c++ << "]part_z_lev" + std::to_string(lev) + "-(m)";
                ofs << m_sep;
                ofs << "[" << c++ << "]part_Ex_lev" + std::to_string(lev) + u_map[FieldProbePIdx::Ex];
                ofs << m_sep;
                ofs << "[" << c++ << "]part_Ey_lev" + std::to_string(lev) + u_map[FieldProbePIdx::Ey];
                ofs << m_sep;
                ofs << "[" << c++ << "]part_Ez_lev" + std::to_string(lev) + u_map[FieldProbePIdx::Ez];
                ofs << m_sep;
                ofs << "[" << c++ << "]part_Bx_lev" + std::to_string(lev) + u_map[FieldProbePIdx::Bx];
                ofs << m_sep;
                ofs << "[" << c++ << "]part_By_lev" + std::to_string(lev) + u_map[FieldProbePIdx::By];
                ofs << m_sep;
                ofs << "[" << c++ << "]part_Bz_lev" + std::to_string(lev) + u_map[FieldProbePIdx::Bz];
                ofs << m_sep;
                ofs << "[" << c++ << "]part_S_lev" + std::to_string(lev) + u_map[FieldProbePIdx::S];

                ofs << std::endl;
                ofs.close();
            }
            // ofs << std::endl;

            // // close file
            // ofs.close();
        }
    }

    m_data_out_level = std::vector<amrex::Vector<amrex::Real>> (nLevel, amrex::Vector<amrex::Real>());
    m_valid_particles_level = std::vector<long> (nLevel, 0);

    //auto series = openPMD::Series(m_path + m_rd_name + ".h5", io::Access::CREATE);
} // end constructor

void FieldProbe::InitData ()
{
    using namespace amrex::literals;

    // create 1D vector for X, Y, and Z coordinates of "particles"
    amrex::Vector<amrex::ParticleReal> xpos;
    amrex::Vector<amrex::ParticleReal> ypos;
    amrex::Vector<amrex::ParticleReal> zpos;

    // for now, only one MPI rank adds probe "particles"
    if (ParallelDescriptor::IOProcessor())
    {
        if (m_probe_geometry == DetectorGeometry::Point)
        {
            xpos.push_back(x_probe);
            ypos.push_back(y_probe);
            zpos.push_back(z_probe);
        }
        else if (m_probe_geometry == DetectorGeometry::Line)
        {
            xpos.reserve(m_resolution);
            ypos.reserve(m_resolution);
            zpos.reserve(m_resolution);

            // Final - initial / steps. Array contains dx, dy, dz
            const amrex::Real DetLineStepSize[3]{
                    (x1_probe - x_probe) / (m_resolution - 1),
                    (y1_probe - y_probe) / (m_resolution - 1),
                    (z1_probe - z_probe) / (m_resolution - 1)};
            for ( int step = 0; step < m_resolution; step++)
            {
                xpos.push_back(x_probe + (DetLineStepSize[0] * step));
                ypos.push_back(y_probe + (DetLineStepSize[1] * step));
                zpos.push_back(z_probe + (DetLineStepSize[2] * step));
            }
        }
        else if (m_probe_geometry == DetectorGeometry::Plane)
        {
            std::size_t const res2 = std::size_t(m_resolution) * std::size_t(m_resolution);
            xpos.reserve(res2);
            ypos.reserve(res2);
            zpos.reserve(res2);

            // ensure that input vectors are normalized
            normalize(target_normal_x, target_normal_y, target_normal_z);
            normalize(target_up_x, target_up_y, target_up_z);

            // create vector orthonormal to input vectors
            const amrex::Real orthotarget[3]{
                target_normal_y * target_up_z - target_normal_z * target_up_y,
                target_normal_z * target_up_x - target_normal_x * target_up_z,
                target_normal_x * target_up_y - target_normal_y * target_up_x};

            // find upper left and lower right bounds of detector
            amrex::Real direction[3]{
                orthotarget[0] - target_up_x,
                orthotarget[1] - target_up_y,
                orthotarget[2] - target_up_z};
            // normalize(direction[0], direction[1], direction[2]);
            normalize(direction[0], direction[1], direction[2]);
            const amrex::Real uppercorner[3]{
                x_probe - (direction[0] * detector_radius),
                y_probe - (direction[1] * detector_radius),
                z_probe - (direction[2] * detector_radius)};
            const amrex::Real lowercorner[3]{
                uppercorner[0] - (target_up_x * std::sqrt(2_rt) * detector_radius),
                uppercorner[1] - (target_up_y * std::sqrt(2_rt) * detector_radius),
                uppercorner[2] - (target_up_z * std::sqrt(2_rt) * detector_radius)};
            const amrex::Real loweropposite[3]{
                x_probe + (direction[0] * detector_radius),
                y_probe + (direction[1] * detector_radius),
                z_probe + (direction[2] * detector_radius)};

            // create array containing point-to-point step size
            const amrex::Real SideStepSize[3]{
                (loweropposite[0] - lowercorner[0]) / (m_resolution - 1),
                (loweropposite[1] - lowercorner[1]) / (m_resolution - 1),
                (loweropposite[2] - lowercorner[2]) / (m_resolution - 1)};
            const amrex::Real UpStepSize[3]{
                (uppercorner[0] - lowercorner[0]) / (m_resolution - 1),
                (uppercorner[1] - lowercorner[1]) / (m_resolution - 1),
                (uppercorner[2] - lowercorner[2]) / (m_resolution - 1)};

            amrex::Real temp_pos[3]{};
            // Starting at the lowercorner point, step sideways and up to form
            // a grid of equally spaced coordinate points
            for ( int sidestep = 0; sidestep < m_resolution; sidestep++)
            {
                for ( int upstep = 0; upstep < m_resolution; upstep++)
                {
                    temp_pos[0] = lowercorner[0] + SideStepSize[0] * sidestep + UpStepSize[0] * upstep;
                    temp_pos[1] = lowercorner[1] + SideStepSize[1] * sidestep + UpStepSize[1] * upstep;
                    temp_pos[2] = lowercorner[2] + SideStepSize[2] * sidestep + UpStepSize[2] * upstep;
#if defined(WARPX_DIM_RZ)
                    auto r = std::sqrt(temp_pos[0] * temp_pos[0] + temp_pos[1] * temp_pos[1]);
                    // if(r > detector_radius) continue;
                    xpos.push_back(r);
                    ypos.push_back(std::atan2(temp_pos[1], temp_pos[0]));
                    zpos.push_back(temp_pos[2]);
#else
                    xpos.push_back(temp_pos[0]);
                    ypos.push_back(temp_pos[1]);
                    zpos.push_back(temp_pos[2]);
#endif
                }
            }
        }
    }
    // add particles on lev 0 to m_probe
    m_probe.AddNParticles(0, xpos, ypos, zpos);

    // Prev lo
    // compute move
    // get a reference to WarpX instance
    //prob_lo_prev = WarpX::GetInstance().Geom(0).ProbLo()[WarpX::moving_window_dir];
    auto & warpx = WarpX::GetInstance();
    prob_lo_prev = warpx.getmoving_window_x();

    if (start_time > 0.0)
    {
        int finest_level = WarpX::GetInstance().finestLevel();
        start_step = static_cast<int>(start_time / WarpX::GetInstance().getdt(finest_level));
        stop_move_step = start_step;
    }
}

void FieldProbe::LoadBalance ()
{
    m_probe.Redistribute();
}

bool FieldProbe::ProbeInDomain () const
{
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();
    int const lev = 0;
    const amrex::Geometry& gm = warpx.Geom(lev);
    const auto *const prob_lo = gm.ProbLo();
    const auto *const prob_hi = gm.ProbHi();

    /*
     * Determine if probe exists within simulation boundaries. During 2D simulations,
     * y values will be set to 0 making it unnecessary to check. Generally, the second
     * value in a position array will be the y value, but in the case of 2D, prob_lo[1]
     * and prob_hi[1] refer to z. This is a result of warpx.Geom(lev).
     */
#if defined(WARPX_DIM_1D_Z)
    return z_probe >= prob_lo[0] && z_probe < prob_hi[0];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return x_probe >= prob_lo[0] && x_probe < prob_hi[0] &&
           z_probe >= prob_lo[1] && z_probe < prob_hi[1];
#else
    return x_probe >= prob_lo[0] && x_probe < prob_hi[0] &&
           y_probe >= prob_lo[1] && y_probe < prob_hi[1] &&
           z_probe >= prob_lo[2] && z_probe < prob_hi[2];
#endif
}

void FieldProbe::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_field_probe_integrate)
    {
        if (!m_intervals.contains(step+1)) { return; }
    }
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // get number of mesh-refinement levels
    const auto nLevel = warpx.finestLevel() + 1;
    //const auto nLevel = warpx.finestLevel() + 1 > max_level + 1 ? max_level + 1 : warpx.finestLevel() + 1;

    m_data_out_level.clear();
    m_valid_particles_level.clear();

    // loop over refinement levels
    for (int lev = 0; lev < nLevel; ++lev)
    {
        amrex::Real const dt = WarpX::GetInstance().getdt(lev);
        // Calculates particle movement in moving window sims
        amrex::Real move_dist = 0.0;
        const int end_moving_window_step = WarpX::end_moving_window_step >= 0 ? WarpX::end_moving_window_step : INT_MAX;
        bool const update_particles_moving_window =
            do_moving_window_FP &&
            step > WarpX::start_moving_window_step &&
            step <= end_moving_window_step &&
            step <= stop_move_step;
        if (update_particles_moving_window)
        {
            const int step_diff = step - m_last_compute_step;
            move_dist = dt*WarpX::moving_window_v*step_diff;
            if(lev == 0){
                // // compute move
                // // get a reference to WarpX instance
                // const auto prob_lo = WarpX::GetInstance().Geom(0).ProbLo()[WarpX::moving_window_dir];
                // amrex::Real probe_move = prob_lo - prob_lo_prev;
                // prob_lo_prev = prob_lo;

                auto & warpx = WarpX::GetInstance();
                // const int moving_dir = WarpX::moving_window_dir;
                const auto prob_lo = warpx.getmoving_window_x();
                amrex::Real probe_move = prob_lo - prob_lo_prev;
                prob_lo_prev = prob_lo;

                move_dist = probe_move;

                const auto temp_warpx_moving_window = WarpX::moving_window_dir;
                if (temp_warpx_moving_window == 0)
                {
                    x_probe += move_dist;
                }
                if (temp_warpx_moving_window == 1)
                {
                    y_probe += move_dist;
                }
                if (temp_warpx_moving_window == WARPX_ZINDEX)
                {
                    z_probe += move_dist;
                }
            }
        }
        // const amrex::MultiFab& (WarpX::*getEfieldFunc) (int lev, int direction);
        // const amrex::MultiFab& (WarpX::*getBfieldFunc) (int lev, int direction);

        // getEfieldFunc = &WarpX::getEfield;
        // getBfieldFunc = &WarpX::getBfield;

        // get MultiFab data at lev
        // const amrex::MultiFab &Ex = (warpx.*getEfieldFunc)(lev, 0);
        // const amrex::MultiFab &Ey = (warpx.*getEfieldFunc)(lev, 1);
        // const amrex::MultiFab &Ez = (warpx.*getEfieldFunc)(lev, 2);
        // const amrex::MultiFab &Bx = (warpx.*getBfieldFunc)(lev, 0);
        // const amrex::MultiFab &By = (warpx.*getBfieldFunc)(lev, 1);
        // const amrex::MultiFab &Bz = (warpx.*getBfieldFunc)(lev, 2);
        const amrex::MultiFab &Ex = warpx.getField(FieldType::Efield_aux, lev, 0);
        const amrex::MultiFab &Ey = warpx.getField(FieldType::Efield_aux, lev, 1);
        const amrex::MultiFab &Ez = warpx.getField(FieldType::Efield_aux, lev, 2);
        const amrex::MultiFab &Bx = warpx.getField(FieldType::Bfield_aux, lev, 0);
        const amrex::MultiFab &By = warpx.getField(FieldType::Bfield_aux, lev, 1);
        const amrex::MultiFab &Bz = warpx.getField(FieldType::Bfield_aux, lev, 2);

        /*
         * Prepare interpolation of field components to probe_position
         * The arrays below store the index type (staggering) of each MultiFab.
         */
        amrex::IndexType const Extype = Ex.ixType();
        amrex::IndexType const Eytype = Ey.ixType();
        amrex::IndexType const Eztype = Ez.ixType();
        amrex::IndexType const Bxtype = Bx.ixType();
        amrex::IndexType const Bytype = By.ixType();
        amrex::IndexType const Bztype = Bz.ixType();

        // loop over each particle
        // TODO: add OMP parallel as in PhysicalParticleContainer::Evolve
        long numparticles = 0; // particles on this MPI rank
        using MyParIter = FieldProbeParticleContainer::iterator;
        for (MyParIter pti(m_probe, lev); pti.isValid(); ++pti)
        {
            // count particle on MPI rank
            numparticles += pti.numParticles();
        }

        if (m_intervals.contains(step+1))
        {
            // reset m_data vector to clear pushed values. Reserves data
            m_data.clear();
            m_data.shrink_to_fit();
            m_data.reserve(numparticles * noutputs);
        }

        for (MyParIter pti(m_probe, lev); pti.isValid(); ++pti)
        {
            const auto getPosition = GetParticlePosition<FieldProbePIdx>(pti);
            auto setPosition = SetParticlePosition<FieldProbePIdx>(pti);

            auto const np = pti.numParticles();
            if (update_particles_moving_window)
            {
                const auto temp_warpx_moving_window = WarpX::moving_window_dir;
                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);
                    if (temp_warpx_moving_window == 0)
                    {
                        setPosition(ip, xp+move_dist, yp, zp);
                    }
                    if (temp_warpx_moving_window == 1)
                    {
                        setPosition(ip, xp, yp+move_dist, zp);
                    }
                    if (temp_warpx_moving_window == WARPX_ZINDEX)
                    {
                        setPosition(ip, xp, yp, zp+move_dist);
                    }
                });
            }
            if( ProbeInDomain() )
            {
                const auto &arrEx = Ex[pti].array();
                const auto &arrEy = Ey[pti].array();
                const auto &arrEz = Ez[pti].array();
                const auto &arrBx = Bx[pti].array();
                const auto &arrBy = By[pti].array();
                const auto &arrBz = Bz[pti].array();

                /*
                 * Make the box cell centered in preparation for the interpolation (and to avoid
                 * including ghost cells in the calculation)
                 */
                amrex::Box box = pti.tilebox();
                box.grow(Ex.nGrowVect());

                //preparing to write data to particle
                auto& attribs = pti.GetStructOfArrays().GetRealData();
                ParticleReal* const AMREX_RESTRICT part_Ex = attribs[FieldProbePIdx::Ex].dataPtr();
                ParticleReal* const AMREX_RESTRICT part_Ey = attribs[FieldProbePIdx::Ey].dataPtr();
                ParticleReal* const AMREX_RESTRICT part_Ez = attribs[FieldProbePIdx::Ez].dataPtr();
                ParticleReal* const AMREX_RESTRICT part_Bx = attribs[FieldProbePIdx::Bx].dataPtr();
                ParticleReal* const AMREX_RESTRICT part_By = attribs[FieldProbePIdx::By].dataPtr();
                ParticleReal* const AMREX_RESTRICT part_Bz = attribs[FieldProbePIdx::Bz].dataPtr();
                ParticleReal* const AMREX_RESTRICT part_S = attribs[FieldProbePIdx::S].dataPtr();

                auto * const AMREX_RESTRICT idcpu = pti.GetStructOfArrays().GetIdCPUData().data();

                const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
                const amrex::XDim3 dinv = WarpX::InvCellSize(lev);
                const Dim3 lo = lbound(box);

                // Temporarily defining modes and interp outside ParallelFor to avoid GPU compilation errors.
                const int temp_modes = WarpX::n_rz_azimuthal_modes;
                const int temp_interp_order = interp_order;
                const bool temp_field_probe_integrate = m_field_probe_integrate;

                // Interpolating to the probe positions for each particle
                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);

                    amrex::ParticleReal Exp = 0._prt, Eyp = 0._prt, Ezp = 0._prt;
                    amrex::ParticleReal Bxp = 0._prt, Byp = 0._prt, Bzp = 0._prt;

                    // first gather E and B to the particle positions
                    doGatherShapeN(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                   arrEx, arrEy, arrEz, arrBx, arrBy, arrBz,
                                   Extype, Eytype, Eztype, Bxtype, Bytype, Bztype,
                                   dinv, xyzmin, lo, temp_modes,
                                   temp_interp_order, false);

                    //Calculate the Poynting Vector S
                    amrex::ParticleReal const sraw[3]{
                        Eyp * Bzp - Ezp * Byp,
                        Ezp * Bxp - Exp * Bzp,
                        Exp * Byp - Eyp * Bxp
                    };
                    amrex::ParticleReal const S = (1._prt / PhysConst::mu0)  * std::sqrt(sraw[0] * sraw[0] + sraw[1] * sraw[1] + sraw[2] * sraw[2]);

                    /*
                     * Determine whether or not to integrate field data.
                     * If not integrating, store instantaneous values.
                     */
                    if (temp_field_probe_integrate)
                    {
                        // store values on particles
                        part_Ex[ip] += Exp * dt; //remember to add lorentz transform
                        part_Ey[ip] += Eyp * dt; //remember to add lorentz transform
                        part_Ez[ip] += Ezp * dt; //remember to add lorentz transform
                        part_Bx[ip] += Bxp * dt; //remember to add lorentz transform
                        part_By[ip] += Byp * dt; //remember to add lorentz transform
                        part_Bz[ip] += Bzp * dt; //remember to add lorentz transform
                        part_S[ip] += S * dt; //remember to add lorentz transform
                    }
                    else
                    {
                        part_Ex[ip] = Exp; //remember to add lorentz transform
                        part_Ey[ip] = Eyp; //remember to add lorentz transform
                        part_Ez[ip] = Ezp; //remember to add lorentz transform
                        part_Bx[ip] = Bxp; //remember to add lorentz transform
                        part_By[ip] = Byp; //remember to add lorentz transform
                        part_Bz[ip] = Bzp; //remember to add lorentz transform
                        part_S[ip] = S; //remember to add lorentz transform
                    }
                });// ParallelFor Close
                // this check is here because for m_field_probe_integrate == True, we always compute
                // but we only write when we truly are in an output interval step
                if (m_intervals.contains(step+1) && np > 0)
                {
                    // This could be optimized by using shared memory.
                    amrex::Gpu::DeviceVector<amrex::Real> dv(np*noutputs);
                    amrex::Real* dvp = dv.data();
                    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
                    {
                        amrex::ParticleReal xp, yp, zp;
                        getPosition(ip, xp, yp, zp);
                        long idx = ip*noutputs;
                        dvp[idx++] = amrex::ParticleIDWrapper{idcpu[ip]};  // all particles created on IO cpu
                        dvp[idx++] = xp;
                        dvp[idx++] = yp;
                        dvp[idx++] = zp;
                        dvp[idx++] = part_Ex[ip];
                        dvp[idx++] = part_Ey[ip];
                        dvp[idx++] = part_Ez[ip];
                        dvp[idx++] = part_Bx[ip];
                        dvp[idx++] = part_By[ip];
                        dvp[idx++] = part_Bz[ip];
                        dvp[idx++] = part_S[ip];
                    });
                    auto oldsize = m_data.size();
                    m_data.resize(oldsize + dv.size());
                    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                                          dv.begin(), dv.end(), &m_data[oldsize]);
                    Gpu::streamSynchronize();
                /* m_data now contains up-to-date values for:
                 *  [x, y, z, Ex, Ey, Ez, Bx, By, Bz, and S] */
                }
            }
        } // end particle iterator loop

        if (m_intervals.contains(step+1))
        {
            // returns total number of mpi notes into mpisize
            const int mpisize = ParallelDescriptor::NProcs();

            // allocates data space for length_array. Will contain size of m_data from each processor
            amrex::Vector<int> length_vector;
            amrex::Vector<int> localsize;

            if (amrex::ParallelDescriptor::IOProcessor()) {
                length_vector.resize(mpisize, 0);
            }
            localsize.resize(1, static_cast<int>(m_data.size()));

            // gather size of m_data from each processor
            amrex::ParallelDescriptor::Gather(localsize.data(), 1,
                                              length_vector.data(), 1,
                                              amrex::ParallelDescriptor::IOProcessorNumber());

            // IO processor sums values from length_array to get size of total output array.
            /* displs records the size of each m_data as well as previous displs. This array
             * tells Gatherv where in the m_data_out array allocation to write incoming data. */
            long total_data_size = 0;
            amrex::Vector<int> displs_vector;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                displs_vector.assign(mpisize, 0);
                total_data_size += length_vector[0];
                for (int i=1; i<mpisize; i++) {
                    displs_vector[i] = (displs_vector[i-1] + length_vector[i-1]);
                    total_data_size += length_vector[i];
                }
                // valid particles are counted (for all MPI ranks) to inform output processes as to size of output
                m_valid_particles = total_data_size / noutputs;
                m_valid_particles_level[lev] = m_valid_particles;
                m_data_out.resize(total_data_size, 0);
            }
            // resize receive buffer (resize, initialize 0)
            // gather m_data of varied lengths from all processors. Prints to m_data_out
            amrex::ParallelDescriptor::Gatherv(m_data.data(), localsize[0],
                                               m_data_out.data(), length_vector, displs_vector,
                                               amrex::ParallelDescriptor::IOProcessorNumber());

            m_data_out_level[lev] = m_data_out;
        }
    }// end loop over refinement levels
    // make sure data is in m_data on the IOProcessor
    // TODO: In the future, we want to use a parallel I/O method instead (plotfiles or openPMD)
    m_last_compute_step = step;
} // end void FieldProbe::ComputeDiags

void FieldProbe::WriteToFile (int step) const
{
    if (!(ProbeInDomain() && amrex::ParallelDescriptor::IOProcessor())) { return; }
    if (!(step >= start_step - 1 && step <= stop_step)) return;

    auto & warpx = WarpX::GetInstance();
    const auto nLevel = warpx.finestLevel() + 1;
    // const auto nLevel = warpx.finestLevel() + 1 > max_level + 1 ? max_level + 1 : warpx.finestLevel() + 1;
    // const auto max_nLevel = nLevel;
    const auto max_nLevel = 1;
    for(int cur_lev = 0; cur_lev < max_nLevel; cur_lev++){

        // if(m_valid_particles_level[cur_lev] == 0) continue;

        // open file
        auto filename = cur_lev > 0 ? m_path + m_rd_name + "_lvl_" + std::to_string(cur_lev) + "." + m_extension:
            m_path + m_rd_name + "." + m_extension;
        std::ofstream ofs{filename, std::ofstream::out | std::ofstream::app};

        long np = 0, data_size = 0;
        // loop over num valid particles to find the lowest particle ID for later sorting
        auto first_id = LONG_MAX; //static_cast<long int>(m_data_out_level[cur_lev][0]);
        for(int lev = cur_lev; lev < nLevel; lev++){
            for (long int i = 0; i < m_valid_particles_level[lev]; i++)
            {
                if (m_data_out_level[lev][i*noutputs] < first_id) {
                    first_id = static_cast<long int>(m_data_out_level[lev][i*noutputs]);
                }
            }
            np += static_cast<long> (m_valid_particles_level[lev]);
            data_size += m_data_out_level[lev].size();
        }
        std::vector<amrex::Real> sorted_data(data_size, 0.0);
        for(int lev = cur_lev; lev < nLevel; lev++){
            // Create a new array to store probe data ordered by id, which will be printed to file.
            // amrex::Vector<amrex::Real> sorted_data;
            // sorted_data.resize(m_data_out_level[lev].size());

            // loop over num valid particles and write data into the appropriately
            // sorted location
            for (long int i = 0; i < m_valid_particles_level[lev]; i++)
            {
                const long int idx = static_cast<long int>(m_data_out_level[lev][i*noutputs]) - first_id;
                for (long int k = 0; k < noutputs; k++)
                {
                    sorted_data[idx * noutputs + k] = m_data_out_level[lev][i * noutputs + k];
                    // sorted_data[i * noutputs + k] = m_data_out_level[lev][i * noutputs + k];
                }
            }
        }
            // // Create a new array to store probe data ordered by id, which will be printed to file.
            // amrex::Vector<amrex::Real> sorted_data;
            // sorted_data.resize(m_data_out_level[lev].size());
            // // push back idx
            // std::vector<long int> idx_vec(m_valid_particles_level[lev]);
            // std::iota(idx_vec.begin(), idx_vec.end(), 0);
            // // sort idx as id number(m_data_out_level[lev][i1*noutputs]) order
            // std::sort (idx_vec.begin(), idx_vec.end(), [&](int i1,int i2){
            //     return static_cast<long int>(m_data_out_level[lev][i1*noutputs]) <
            //         static_cast<long int>(m_data_out_level[lev][i2*noutputs]);
            // });
            // // push back data
            // for (long int i = 0; i < m_valid_particles_level[lev]; i++)
            // {
            //     for (long int k = 0; k < noutputs; k++)
            //     {
            //         // sorted_data[idx * noutputs + k] = m_data_out_level[lev][i * noutputs + k];
            //         sorted_data[i * noutputs + k] = m_data_out_level[lev][idx_vec[i] * noutputs + k];
            //     }
            // }


        // loop over num valid particles and write
        for (long int i = 0; i < np; i++)
        {
            ofs << std::fixed << std::defaultfloat;
            ofs << step + 1;
            ofs << m_sep;
            ofs << std::fixed << std::setprecision(14) << std::scientific;
            // write time
            ofs << WarpX::GetInstance().gett_new(0);

            // start at k = 1 since the particle id is not written to file
            for (long int k = 1; k < noutputs; k++)
            {
                ofs << m_sep;
                ofs << sorted_data[i * noutputs + k];
            }
            ofs << std::endl;
        } // end loop over data size
        // close file
        ofs.close();
        // if(cur_lev == 0) this->WriteToFileOpenPMD(step, sorted_data);
    }
}

void FieldProbe::WriteToFileOpenPMD (int step, std::vector<amrex::Real> sorted_data) const
{
    // if (!(ProbeInDomain() && amrex::ParallelDescriptor::IOProcessor())) return;
    // if (!(step >= start_step - 1 && step <= stop_step)) return;

    //m_Series = std::make_unique<openPMD::Series>(m_path + "/" + std::to_string(step) + "/" + m_rd_name + ".h5", openPMD::Access::CREATE);
    auto m_Series = openPMD::Series(m_path + "/h5dump/" + m_rd_name + "_" + std::to_string(step + 1) + ".h5", openPMD::Access::CREATE);

    // int data_size = sorted_data.size();
    int np = 0;

    auto & warpx = WarpX::GetInstance();
    const auto nLevel = warpx.finestLevel() + 1;
    for(int lev = 0; lev < nLevel; lev++){
        //np += static_cast<unsigned long> (m_valid_particles_level[lev]);
        np += static_cast<int> (m_valid_particles_level[lev]);
    }

    openPMD::Iteration currIteration = m_Series.iterations[step+1];
    openPMD::ParticleSpecies currSpecies = currIteration.particles["species"];
    currIteration.setTime(WarpX::GetInstance().gett_new(0));

    const std::shared_ptr<float> curr(
        new float[np], [](float const *p) { 
            delete[] p; 
            p = nullptr;
        }
    );
    openPMD::Datatype dtype = openPMD::determineDatatype(curr);
    auto d = openPMD::Dataset(dtype, {(uint64_t) np});
    // std::string options = "{}";
    // auto realType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), openPMD::Extent{np}); //, options);
    // auto idType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), openPMD::Extent{np}); //, options);

    std::vector<std::string> const positionComponents = {"x", "y", "z"};

    for(auto const& comp : positionComponents) {
        currSpecies["position"][comp].resetDataset(d);
        currSpecies["E"][comp].resetDataset(d);
        currSpecies["B"][comp].resetDataset(d);
    }
    // auto const scalar = openPMD::RecordComponent::SCALAR;
    currSpecies["S"]["S"].resetDataset(d);
    currSpecies["id"]["id"].resetDataset(d);

    m_Series.flush();

    std::vector<std::string> const components = {"position", "E", "B", "S", "id"};

    for(int idx = 0; idx < components.size(); idx++){
        if(idx < 3){
            for (auto currDim = 0; currDim < 3; currDim++) {
                for (auto i = 0; i < static_cast<int> (np); i++) {
                    int arg_offset = 1 + idx * 3 + currDim;
                    curr.get()[i] = static_cast<float> (sorted_data[i * noutputs + arg_offset]);
                }
                //unsigned long long offset = 0;
                uint64_t offset = 0;
                currSpecies[components[idx]][positionComponents[currDim]]
                    .storeChunk(curr, openPMD::Offset{offset}, openPMD::Extent{(uint64_t) np});
                m_Series.flush();
            }
        }else{
            for (int i = 0; i < static_cast<int> (np); i++) {
                int arg_offset;
                if(idx == 3){
                    arg_offset = 10;
                }else{
                    arg_offset = 0;
                }
                curr.get()[i] = static_cast<float> (sorted_data[i * noutputs + arg_offset]);
            }
            // unsigned long long offset = 0;
            uint64_t offset = 0;
            currSpecies[components[idx]][components[idx]]
                .storeChunk(curr, openPMD::Offset{offset}, openPMD::Extent{(uint64_t) np});
            m_Series.flush();
        }
    }
    m_Series.flush();
    currIteration.close();
    m_Series.close();

    ////////////////////////////////////////////

    // auto & warpx = WarpX::GetInstance();
    // const auto nLevel = warpx.finestLevel() + 1;
    // // unsigned long np = 0;
    // int np = 0;
    // for(int lev = 0; lev < nLevel; lev++){
    //     //np += static_cast<unsigned long> (m_valid_particles_level[lev]);
    //     np += static_cast<int> (m_valid_particles_level[lev]);
    // }

    // //openPMD::Iteration currIteration = m_Series->writeIterations()[step + 1];

    // openPMD::Iteration currIteration = m_Series.iterations[step+1];
    // openPMD::ParticleSpecies currSpecies = currIteration.particles["species"];

    // const std::shared_ptr<float> curr(
    //     new float[np], [](float const *p) { 
    //         delete[] p; 
    //         p = nullptr;
    //     }
    // );
    // openPMD::Datatype dtype = openPMD::determineDatatype(curr);
    // auto d = openPMD::Dataset(dtype, {(uint64_t) np});
    // // std::string options = "{}";
    // // auto realType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), openPMD::Extent{np}); //, options);
    // // auto idType = openPMD::Dataset(openPMD::determineDatatype<amrex::ParticleReal>(), openPMD::Extent{np}); //, options);

    // std::vector<std::string> const positionComponents = {"x", "y", "z"};

    // for(auto const& comp : positionComponents) {
    //     currSpecies["position"][comp].resetDataset(d);
    //     currSpecies["E"][comp].resetDataset(d);
    //     currSpecies["B"][comp].resetDataset(d);
    // }
    // // auto const scalar = openPMD::RecordComponent::SCALAR;
    // currSpecies["S"]["S"].resetDataset(d);
    // currSpecies["id"]["id"].resetDataset(d);

    // m_Series.flush();

    // std::vector<std::string> const components = {"position", "E", "B", "S", "id"};
    // for(int lev = 0; lev < nLevel; lev++){
    //     for(int idx = 0; idx < components.size(); idx++){
    //         if(idx < 3){
    //             for (auto currDim = 0; currDim < 3; currDim++) {
    //                 for (auto i = 0; i < static_cast<int> (np); i++) {
    //                     int arg_offset = 1 + idx * 3 + currDim;
    //                     curr.get()[i] = static_cast<float> (m_data_out_level[lev][i * noutputs + arg_offset]);
    //                 }
    //                 //unsigned long long offset = 0;
    //                 uint64_t offset = 0;
    //                 currSpecies[components[idx]][positionComponents[currDim]]
    //                     .storeChunk(curr, openPMD::Offset{offset}, openPMD::Extent{(uint64_t) np});
    //                 m_Series.flush();
    //             }
    //         }else{
    //             for (int i = 0; i < static_cast<int> (np); i++) {
    //                 int arg_offset;
    //                 if(idx == 3){
    //                     arg_offset = 10;
    //                 }else{
    //                     arg_offset = 0;
    //                 }
    //                 curr.get()[i] = static_cast<float> (m_data_out_level[lev][i * noutputs + arg_offset]);
    //             }
    //             // unsigned long long offset = 0;
    //             uint64_t offset = 0;
    //             currSpecies[components[idx]][components[idx]]
    //                 .storeChunk(curr, openPMD::Offset{offset}, openPMD::Extent{(uint64_t) np});
    //             m_Series.flush();
    //         }
    //     }
    // }
    // m_Series.flush();
    // currIteration.close();
    // m_Series.close();
}
