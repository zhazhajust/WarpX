#include "FourierRadiationDiag.H"

#include "Diagnostics/FourierRadiation.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>

#include <fstream>
#include <iomanip>
#include <vector>

using namespace amrex;

FourierRadiationDiag::FourierRadiationDiag (std::string const& rd_name)
    : ReducedDiags{rd_name}
{
    auto const* radiation = WarpX::GetInstance().GetFourierRadiation();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        radiation != nullptr && radiation->Enabled(),
        "FourierRadiation reduced diagnostic was not initialized. Check "
        + rd_name + ".species/frequencies/theta/phi.");

    m_data.resize(radiation->NumGridNodes(), 0._rt);

    if (ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
        ofs << "# step time frequency_index frequency_Hz theta_index phi_index "
               "d2I_domega_dOmega_SI\n";
    }
}

void
FourierRadiationDiag::ComputeDiags (int step)
{
    if (!m_intervals.contains(step+1)) {
        return;
    }

    auto* radiation = WarpX::GetInstance().GetFourierRadiation();
    radiation->GetIntensity(m_data);
}

void
FourierRadiationDiag::WriteToFile (int step) const
{
    auto* radiation = WarpX::GetInstance().GetFourierRadiation();
    if (radiation == nullptr || !radiation->Enabled()) {
        return;
    }

    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
            std::ofstream::out | std::ofstream::app};
        ofs << std::fixed << std::setprecision(m_precision) << std::scientific;

        int const n_omega = radiation->NumOmega();
        int const n_theta = radiation->NumTheta();
        int const n_phi = radiation->NumPhi();
        Real const time = WarpX::GetInstance().gett_new(0);

        for (int iphi = 0; iphi < n_phi; ++iphi) {
            for (int itheta = 0; itheta < n_theta; ++itheta) {
                for (int iomega = 0; iomega < n_omega; ++iomega) {
                    int const idx = iomega + n_omega * (itheta + n_theta * iphi);
                    ofs << step+1 << m_sep << time << m_sep
                        << iomega << m_sep << radiation->Frequency(iomega) << m_sep
                        << itheta << m_sep << iphi << m_sep << m_data[idx] << "\n";
                }
            }
        }
    }

    if (radiation->ResetAfterOutput()) {
        radiation->Reset();
    }
}
