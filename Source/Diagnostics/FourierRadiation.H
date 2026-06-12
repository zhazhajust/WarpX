#ifndef WARPX_DIAGNOSTICS_FOURIERRADIATION_H_
#define WARPX_DIAGNOSTICS_FOURIERRADIATION_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct FourierRadiationParticleRecord
{
    amrex::Real x_mid = 0.0;
    amrex::Real y_mid = 0.0;
    amrex::Real z_mid = 0.0;
    amrex::Real ux_old = 0.0;
    amrex::Real uy_old = 0.0;
    amrex::Real uz_old = 0.0;
    amrex::Real ux_new = 0.0;
    amrex::Real uy_new = 0.0;
    amrex::Real uz_new = 0.0;
    amrex::Real weight = 0.0;
};

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
    void AddLocalParticleRecords (
        FourierRadiationParticleRecord const* records,
        long num_records);
    void ClearLocalParticleRecords ();
    void ComputeGlobalWorkQueue (
        amrex::Real dt,
        amrex::Real charge,
        amrex::Real radiation_time);

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
    [[nodiscard]] bool DoParticleFilter () const { return m_do_particle_filter; }
    [[nodiscard]] bool ParticleFilterIsLatched () const { return m_particle_filter_is_latched; }
    [[nodiscard]] amrex::ParserExecutor<8> ParticleFilterFunction () const
    {
        return m_particle_filter_function;
    }
    [[nodiscard]] amrex::Real ParticleFraction () const { return m_particle_fraction; }

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
    amrex::Real m_particle_fraction = 1.0;
    bool m_do_particle_filter = false;
    bool m_particle_filter_is_latched = false;
    std::string m_omega_grid = "linear";
    std::unique_ptr<amrex::Parser> m_particle_filter_parser;
    amrex::ParserExecutor<8> m_particle_filter_function;

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

    std::vector<FourierRadiationParticleRecord> m_local_particle_records;
    std::mutex m_local_particle_records_mutex;
};

#endif
