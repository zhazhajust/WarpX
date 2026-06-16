/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "TimeDomainRadiationDiag.H"

#include "Diagnostics/TimeDomainRadiation.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>

#include <fstream>
#include <iomanip>

TimeDomainRadiationDiag::TimeDomainRadiationDiag (std::string const& rd_name)
    : ReducedDiags{rd_name}
{
    auto const* radiation = WarpX::GetInstance().GetTimeDomainRadiation();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        radiation != nullptr && radiation->Enabled(),
        "TimeDomainRadiation reduced diagnostic was not initialized. Check "
        "warpx.reduced_diags_names and <diag>.type.");

    m_data.resize(static_cast<std::size_t>(radiation->NumComponents())
        * radiation->NumGridNodes(), 0._rt);

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
        ofs << "#[0]step()" << m_sep << "[1]time(s)" << m_sep
            << "[2]time_index()" << m_sep << "[3]radiation_time(s)" << m_sep
            << "[4]theta(rad)" << m_sep << "[5]phi(rad)" << m_sep
            << "[6]Ex(V/m)" << m_sep << "[7]Ey(V/m)" << m_sep << "[8]Ez(V/m)\n";
    }
}

void
TimeDomainRadiationDiag::ComputeDiags (int step)
{
    if (!m_intervals.contains(step+1)) { return; }

    auto* radiation = WarpX::GetInstance().GetTimeDomainRadiation();
    radiation->AddToHostVector(m_data);
    if (radiation->ResetAfterOutput()) {
        radiation->Reset();
    }
}

void
TimeDomainRadiationDiag::WriteToFile (int step) const
{
    auto const* radiation = WarpX::GetInstance().GetTimeDomainRadiation();
    if (radiation == nullptr || !radiation->Enabled()) {
        return;
    }

    auto const& warpx = WarpX::GetInstance();
    amrex::Real const sim_time = warpx.gett_new(0);

    std::ofstream ofs;
    ofs.open(m_path + m_rd_name + "." + m_extension, std::ofstream::out | std::ofstream::app);
    ofs << std::setprecision(m_precision);

    int const n_time = radiation->NumTime();
    int const n_theta = radiation->NumTheta();
    int const n_phi = radiation->NumPhi();
    int const num_grid_nodes = radiation->NumGridNodes();

    for (int it = 0; it < n_time; ++it) {
        for (int ith = 0; ith < n_theta; ++ith) {
            for (int iph = 0; iph < n_phi; ++iph) {
                int const node = (it * n_theta + ith) * n_phi + iph;
                ofs << step << m_sep << sim_time << m_sep
                    << it << m_sep << radiation->Time(it) << m_sep
                    << radiation->Theta(ith) << m_sep << radiation->Phi(iph) << m_sep
                    << m_data[node] << m_sep
                    << m_data[num_grid_nodes + node] << m_sep
                    << m_data[2*num_grid_nodes + node] << "\n";
            }
        }
    }
}
