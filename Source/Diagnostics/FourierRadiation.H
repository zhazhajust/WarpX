#ifndef WARPX_DIAGNOSTICS_FOURIERRADIATION_H_
#define WARPX_DIAGNOSTICS_FOURIERRADIATION_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

class FourierRadiation
{
public:
    FourierRadiation ();

    [[nodiscard]] bool Enabled () const { return m_enabled; }
    [[nodiscard]] bool IsSpeciesSelected (std::string const& species_name) const;
    [[nodiscard]] int NumGridNodes () const { return m_num_grid_nodes; }

    [[nodiscard]] amrex::Real const* OmegaData () const { return m_omega_2pi.dataPtr(); }
    [[nodiscard]] amrex::Real const* SinThetaData () const { return m_sin_theta.dataPtr(); }
    [[nodiscard]] amrex::Real const* CosThetaData () const { return m_cos_theta.dataPtr(); }
    [[nodiscard]] amrex::Real const* SinPhiData () const { return m_sin_phi.dataPtr(); }
    [[nodiscard]] amrex::Real const* CosPhiData () const { return m_cos_phi.dataPtr(); }

    [[nodiscard]] amrex::Real* AmpXReData () { return m_amp_x_re.dataPtr(); }
    [[nodiscard]] amrex::Real* AmpXImData () { return m_amp_x_im.dataPtr(); }
    [[nodiscard]] amrex::Real* AmpYReData () { return m_amp_y_re.dataPtr(); }
    [[nodiscard]] amrex::Real* AmpYImData () { return m_amp_y_im.dataPtr(); }
    [[nodiscard]] amrex::Real* AmpZReData () { return m_amp_z_re.dataPtr(); }
    [[nodiscard]] amrex::Real* AmpZImData () { return m_amp_z_im.dataPtr(); }

    void GetIntensity (std::vector<amrex::Real>& intensity) const;
    void Reset ();

    [[nodiscard]] int NumOmega () const { return m_num_omega; }
    [[nodiscard]] int NumTheta () const { return m_num_theta; }
    [[nodiscard]] int NumPhi () const { return m_num_phi; }
    [[nodiscard]] amrex::Real OmegaMin () const { return m_omega_min; }
    [[nodiscard]] amrex::Real OmegaMax () const { return m_omega_max; }
    [[nodiscard]] amrex::Real Frequency (int i) const { return m_frequency[i]; }
    [[nodiscard]] amrex::Real ThetaMin () const { return m_theta_min; }
    [[nodiscard]] amrex::Real ThetaMax () const { return m_theta_max; }
    [[nodiscard]] amrex::Real PhiMin () const { return m_phi_min; }
    [[nodiscard]] amrex::Real PhiMax () const { return m_phi_max; }
    [[nodiscard]] bool ResetAfterOutput () const { return m_reset_after_output; }

private:
    void Allocate ();

    bool m_enabled = false;
    bool m_reset_after_output = false;
    std::string m_species_name;

    int m_num_omega = 0;
    int m_num_theta = 0;
    int m_num_phi = 0;
    int m_num_grid_nodes = 0;

    amrex::Real m_omega_min = 0.0;
    amrex::Real m_omega_max = 0.0;
    amrex::Real m_theta_min = 0.0;
    amrex::Real m_theta_max = 0.0;
    amrex::Real m_phi_min = 0.0;
    amrex::Real m_phi_max = 0.0;
    std::string m_omega_grid = "linear";

    std::vector<amrex::Real> m_frequency;
    amrex::Gpu::DeviceVector<amrex::Real> m_omega_2pi;
    amrex::Gpu::DeviceVector<amrex::Real> m_sin_theta;
    amrex::Gpu::DeviceVector<amrex::Real> m_cos_theta;
    amrex::Gpu::DeviceVector<amrex::Real> m_sin_phi;
    amrex::Gpu::DeviceVector<amrex::Real> m_cos_phi;

    amrex::Gpu::DeviceVector<amrex::Real> m_amp_x_re;
    amrex::Gpu::DeviceVector<amrex::Real> m_amp_x_im;
    amrex::Gpu::DeviceVector<amrex::Real> m_amp_y_re;
    amrex::Gpu::DeviceVector<amrex::Real> m_amp_y_im;
    amrex::Gpu::DeviceVector<amrex::Real> m_amp_z_re;
    amrex::Gpu::DeviceVector<amrex::Real> m_amp_z_im;
};

#endif
