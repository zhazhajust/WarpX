/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_TIMEDOMAINRADIATION_K_H_
#define WARPX_DIAGNOSTICS_TIMEDOMAINRADIATION_K_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

struct TimeDomainRadiationDeviceData
{
    amrex::Real* field = nullptr;
    amrex::Real const* sin_theta = nullptr;
    amrex::Real const* cos_theta = nullptr;
    amrex::Real const* sin_phi = nullptr;
    amrex::Real const* cos_phi = nullptr;
    int n_time = 0;
    int n_theta = 0;
    int n_phi = 0;
    int n_components = 3;
    amrex::Real time_min = amrex::Real(0);
    amrex::Real time_max = amrex::Real(0);
    amrex::Real dt_screen = amrex::Real(1);
    amrex::Real radius = amrex::Real(1);
    bool far_field_approx = true;
};

namespace warpx::diagnostics::time_domain_radiation
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real component_offset (
        TimeDomainRadiationDeviceData const& data,
        int const component,
        int const it,
        int const ith,
        int const iph) noexcept
    {
        return static_cast<amrex::Real>(
            (((component * data.n_time + it) * data.n_theta + ith) * data.n_phi + iph));
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void deposit_segment (
        TimeDomainRadiationDeviceData const& data,
        amrex::Real time_ret,
        amrex::Real time_ret_prev,
        amrex::GpuArray<amrex::Real, 3> const& far_field,
        int const ith,
        int const iph) noexcept
    {
        if (time_ret < time_ret_prev) {
            amrex::Real tmp = time_ret;
            time_ret = time_ret_prev;
            time_ret_prev = tmp;
        }

        if (time_ret_prev < data.time_min || time_ret > data.time_max) {
            return;
        }

        int it_prev = static_cast<int>((time_ret_prev - data.time_min) / data.dt_screen);
        int it = static_cast<int>((time_ret - data.time_min) / data.dt_screen);
        if (it_prev < 0 || it < 0 || it_prev >= data.n_time || it >= data.n_time) {
            return;
        }

        for (int idx = it_prev; idx < it; ++idx) {
            amrex::Real const next_time = data.time_min + (idx + 1) * data.dt_screen;
            amrex::Real const weight = next_time - time_ret_prev;
            for (int c = 0; c < data.n_components; ++c) {
                amrex::Gpu::Atomic::AddNoRet(
                    data.field + static_cast<long>(
                        (((c * data.n_time + idx) * data.n_theta + ith) * data.n_phi + iph)),
                    weight * far_field[c]);
            }
            time_ret_prev = next_time;
        }

        amrex::Real const weight = time_ret - time_ret_prev;
        for (int c = 0; c < data.n_components; ++c) {
            amrex::Gpu::Atomic::AddNoRet(
                data.field + static_cast<long>(
                    (((c * data.n_time + it) * data.n_theta + ith) * data.n_phi + iph)),
                weight * far_field[c]);
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void accumulate_particle (
        TimeDomainRadiationDeviceData const& data,
        amrex::Real const x,
        amrex::Real const y,
        amrex::Real const z,
        amrex::Real const x_prev,
        amrex::Real const y_prev,
        amrex::Real const z_prev,
        amrex::Real const ux,
        amrex::Real const uy,
        amrex::Real const uz,
        amrex::Real const ux_prev,
        amrex::Real const uy_prev,
        amrex::Real const uz_prev,
        amrex::Real const time,
        amrex::Real const dt,
        amrex::Real const charge) noexcept
    {
        amrex::Real const gamma = std::sqrt(amrex::Real(1) + ux*ux + uy*uy + uz*uz);
        amrex::Real const gamma_prev =
            std::sqrt(amrex::Real(1) + ux_prev*ux_prev + uy_prev*uy_prev + uz_prev*uz_prev);
        amrex::GpuArray<amrex::Real, 3> const beta{
            ux / gamma, uy / gamma, uz / gamma};
        amrex::GpuArray<amrex::Real, 3> const beta_prev{
            ux_prev / gamma_prev, uy_prev / gamma_prev, uz_prev / gamma_prev};
        amrex::GpuArray<amrex::Real, 3> const beta_dot{
            (beta[0] - beta_prev[0]) / dt,
            (beta[1] - beta_prev[1]) / dt,
            (beta[2] - beta_prev[2]) / dt};
        amrex::Real const time_prev = time - dt;

        for (int ith = 0; ith < data.n_theta; ++ith) {
            amrex::Real const sin_theta = data.sin_theta[ith];
            amrex::Real const cos_theta = data.cos_theta[ith];
            for (int iph = 0; iph < data.n_phi; ++iph) {
                amrex::GpuArray<amrex::Real, 3> n{
                    sin_theta * data.cos_phi[iph],
                    sin_theta * data.sin_phi[iph],
                    cos_theta};
                amrex::Real radius = data.radius;
                amrex::Real time_ret = time - (n[0]*x + n[1]*y + n[2]*z) / PhysConst::c;
                amrex::Real time_ret_prev =
                    time_prev - (n[0]*x_prev + n[1]*y_prev + n[2]*z_prev) / PhysConst::c;

                if (!data.far_field_approx) {
                    amrex::GpuArray<amrex::Real, 3> r{
                        data.radius*n[0] - x,
                        data.radius*n[1] - y,
                        data.radius*n[2] - z};
                    amrex::Real const rnorm =
                        std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
                    time_ret = time + (rnorm - data.radius) / PhysConst::c;

                    r = amrex::GpuArray<amrex::Real, 3>{
                        data.radius*n[0] - x_prev,
                        data.radius*n[1] - y_prev,
                        data.radius*n[2] - z_prev};
                    radius = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
                    n = amrex::GpuArray<amrex::Real, 3>{
                        r[0] / radius, r[1] / radius, r[2] / radius};
                    time_ret_prev = time_prev + (radius - data.radius) / PhysConst::c;
                }

                amrex::Real const one_minus_n_dot_beta =
                    amrex::Real(1) -
                    (n[0]*beta_prev[0] + n[1]*beta_prev[1] + n[2]*beta_prev[2]);
                amrex::Real const denom =
                    one_minus_n_dot_beta * one_minus_n_dot_beta * one_minus_n_dot_beta
                    * radius;
                if (denom == amrex::Real(0)) {
                    continue;
                }

                amrex::GpuArray<amrex::Real, 3> const a{
                    n[0] - beta_prev[0],
                    n[1] - beta_prev[1],
                    n[2] - beta_prev[2]};
                amrex::GpuArray<amrex::Real, 3> const b{
                    a[1]*beta_dot[2] - a[2]*beta_dot[1],
                    a[2]*beta_dot[0] - a[0]*beta_dot[2],
                    a[0]*beta_dot[1] - a[1]*beta_dot[0]};
                amrex::Real const si_prefactor =
                    amrex::Real(1) /
                    (amrex::Real(4) * MathConst::pi * PhysConst::epsilon_0 * PhysConst::c);
                amrex::GpuArray<amrex::Real, 3> const far_field{
                    si_prefactor * charge * (n[1]*b[2] - n[2]*b[1]) / denom,
                    si_prefactor * charge * (n[2]*b[0] - n[0]*b[2]) / denom,
                    si_prefactor * charge * (n[0]*b[1] - n[1]*b[0]) / denom};

                deposit_segment(data, time_ret, time_ret_prev, far_field, ith, iph);
            }
        }
    }
}

#endif
