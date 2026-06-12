#ifndef WARPX_DIAGNOSTICS_FOURIERRADIATION_K_H_
#define WARPX_DIAGNOSTICS_FOURIERRADIATION_K_H_

#include "Utils/WarpXConst.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

struct FourierRadiationDeviceData
{
    amrex::Real const* omega_2pi = nullptr;
    amrex::Real const* sin_theta = nullptr;
    amrex::Real const* cos_theta = nullptr;
    amrex::Real const* sin_phi = nullptr;
    amrex::Real const* cos_phi = nullptr;
    amrex::Real* amp = nullptr;
    int num_omega = 0;
    int num_theta = 0;
    int num_phi = 0;
    int num_grid_nodes = 0;
    amrex::Real particle_fraction = 1.0;
};

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

struct FourierRadiationAmplitude
{
    amrex::Real x_re = 0.0;
    amrex::Real x_im = 0.0;
    amrex::Real y_re = 0.0;
    amrex::Real y_im = 0.0;
    amrex::Real z_re = 0.0;
    amrex::Real z_im = 0.0;
};

namespace warpx::diagnostics::fourier_radiation
{
    static constexpr int n_amp_components = 6;
    static constexpr int amp_x_re = 0;
    static constexpr int amp_x_im = 1;
    static constexpr int amp_y_re = 2;
    static constexpr int amp_y_im = 3;
    static constexpr int amp_z_re = 4;
    static constexpr int amp_z_im = 5;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real const* grid_omega (FourierRadiationDeviceData const& data) noexcept
    {
        return data.omega_2pi;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real const* grid_sin_theta (FourierRadiationDeviceData const& data) noexcept
    {
        return data.sin_theta;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real const* grid_cos_theta (FourierRadiationDeviceData const& data) noexcept
    {
        return data.cos_theta;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real const* grid_sin_phi (FourierRadiationDeviceData const& data) noexcept
    {
        return data.sin_phi;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real const* grid_cos_phi (FourierRadiationDeviceData const& data) noexcept
    {
        return data.cos_phi;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real* amp_component (
        FourierRadiationDeviceData const& data, int const component) noexcept
    {
        return data.amp + component * data.num_grid_nodes;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    FourierRadiationAmplitude contribution (
        FourierRadiationDeviceData const& data,
        FourierRadiationParticleRecord const& record,
        int const gti,
        amrex::Real const dt,
        amrex::Real const charge,
        amrex::Real const radiation_time) noexcept
    {
        int const i_phi = gti / (data.num_omega * data.num_theta);
        int const i_theta =
            (gti - i_phi * data.num_omega * data.num_theta) / data.num_omega;
        int const i_omega =
            gti - i_phi * data.num_omega * data.num_theta - i_theta * data.num_omega;

        amrex::Real const nx = grid_sin_theta(data)[i_theta] * grid_cos_phi(data)[i_phi];
        amrex::Real const ny = grid_sin_theta(data)[i_theta] * grid_sin_phi(data)[i_phi];
        amrex::Real const nz = grid_cos_theta(data)[i_theta];

        constexpr amrex::Real inv_c = amrex::Real(1.0) / PhysConst::c;
        amrex::Real const dt_inv = amrex::Real(1.0) / dt;
        amrex::Real const gamma_inv_prev = amrex::Math::rsqrt(
            amrex::Real(1.0)
            + (record.ux_old*record.ux_old + record.uy_old*record.uy_old
                     + record.uz_old*record.uz_old) * inv_c * inv_c);
        amrex::Real const gamma_inv_new = amrex::Math::rsqrt(
            amrex::Real(1.0)
            + (record.ux_new*record.ux_new + record.uy_new*record.uy_new
                     + record.uz_new*record.uz_new) * inv_c * inv_c);

        amrex::Real const betax_prev = record.ux_old * inv_c * gamma_inv_prev;
        amrex::Real const betay_prev = record.uy_old * inv_c * gamma_inv_prev;
        amrex::Real const betaz_prev = record.uz_old * inv_c * gamma_inv_prev;
        amrex::Real const betax_new = record.ux_new * inv_c * gamma_inv_new;
        amrex::Real const betay_new = record.uy_new * inv_c * gamma_inv_new;
        amrex::Real const betaz_new = record.uz_new * inv_c * gamma_inv_new;

        amrex::Real const betax = amrex::Real(0.5) * (betax_prev + betax_new);
        amrex::Real const betay = amrex::Real(0.5) * (betay_prev + betay_new);
        amrex::Real const betaz = amrex::Real(0.5) * (betaz_prev + betaz_new);
        amrex::Real const ax = (betax_new - betax_prev) * dt_inv;
        amrex::Real const ay = (betay_new - betay_prev) * dt_inv;
        amrex::Real const az = (betaz_new - betaz_prev) * dt_inv;

        amrex::Real const c2_denom = amrex::Real(1.0) - (betax*nx + betay*ny + betaz*nz);
        if (amrex::Math::abs(c2_denom) <= std::numeric_limits<amrex::Real>::min()) {
            return {};
        }

        amrex::Real const c2 = amrex::Real(1.0) / c2_denom;
        amrex::Real const c1 = (ax*nx + ay*ny + az*nz) * c2 * c2;
        amrex::Real const amplitude_x = c1 * (nx - betax) - c2 * ax;
        amrex::Real const amplitude_y = c1 * (ny - betay) - c2 * ay;
        amrex::Real const amplitude_z = c1 * (nz - betaz) - c2 * az;

        amrex::Real const phase = grid_omega(data)[i_omega]
            * (radiation_time + amrex::Real(0.5)*dt
               - (record.x_mid*nx + record.y_mid*ny + record.z_mid*nz) * inv_c);
        amrex::Real const sin_phase = std::sin(phase);
        amrex::Real const cos_phase = std::cos(phase);
        amrex::Real charge_weight_dt = record.weight * charge * dt;
        if (data.particle_fraction < amrex::Real(1.0)) {
            charge_weight_dt /= data.particle_fraction;
        }

        return FourierRadiationAmplitude{
            charge_weight_dt * amplitude_x * cos_phase,
            charge_weight_dt * amplitude_x * sin_phase,
            charge_weight_dt * amplitude_y * cos_phase,
            charge_weight_dt * amplitude_y * sin_phase,
            charge_weight_dt * amplitude_z * cos_phase,
            charge_weight_dt * amplitude_z * sin_phase};
    }
}

#endif
