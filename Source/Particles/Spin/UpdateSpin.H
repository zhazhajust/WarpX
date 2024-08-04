#ifndef WARPX_SPIN_PUSHER_UPDATESPIN_H_
#define WARPX_SPIN_PUSHER_UPDATESPIN_H_

#include "Utils/WarpXConst.H"
#include <AMReX_REAL.H>
#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <cmath>
// #include <random>

// // Function to initialize a random point on a sphere
// AMREX_GPU_HOST_DEVICE AMREX_INLINE
// void InitSpin(
//     amrex::Real& sx_f, amrex::Real& sy_f, amrex::Real& sz_f,
//     amrex::Real radius = 1.0)
// {
//     // Setup random number generators
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> dis(-radius, radius);
//     std::uniform_real_distribution<> phi_dis(0, 2 * M_PI);

//     // Generate z coordinate
//     amrex::Real z = dis(gen);
//     // Generate azimuthal angle
//     amrex::Real phi = phi_dis(gen);
//     // Calculate sin(theta) based on z
//     amrex::Real sin_theta = std::sin(std::acos(z / radius));
//     // Calculate cartesian coordinates x and y
//     amrex::Real x = sin_theta * std::cos(phi);
//     amrex::Real y = sin_theta * std::sin(phi);

//     // Assign the values to output references
//     sx_f = x;
//     sy_f = y;
//     sz_f = z;
// }

using namespace amrex::literals;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateSpin(
    amrex::ParticleReal& sx_f, amrex::ParticleReal& sy_f, amrex::ParticleReal& sz_f,
    const amrex::ParticleReal ux_i, const amrex::ParticleReal uy_i, const amrex::ParticleReal uz_i,
    const amrex::ParticleReal ux_f, const amrex::ParticleReal uy_f, const amrex::ParticleReal uz_f,
    const amrex::Real Ex, const amrex::Real Ey, const amrex::Real Ez,
    const amrex::Real Bx, const amrex::Real By, const amrex::Real Bz,
    const amrex::Real tauconst, const amrex::Real anom)
{

    const amrex::ParticleReal sx_i = sx_f, sy_i = sy_f, sz_i = sz_f;
    // Momentum at a half-timestep forward using Boris assumption
    amrex::ParticleReal ux = (ux_i + ux_f) / 2._prt;
    amrex::ParticleReal uy = (uy_i + uy_f) / 2._prt;
    amrex::ParticleReal uz = (uz_i + uz_f) / 2._prt;

    // constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    // // Compute inverse Lorentz factor
    // const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);

    // convert to be beta
    ux /= PhysConst::c;
    uy /= PhysConst::c;
    uz /= PhysConst::c;
    // Assume 1./gamma at midpoint can be calculated from (average) momentum at midpoint
    // Compute inverse Lorentz factor
    const amrex::ParticleReal inv_gamma = 1._prt / std::sqrt(1._prt + ux*ux + uy*uy + uz*uz);

    // Define parameter: 1./(1+gamma)
    const amrex::ParticleReal inv_1pgamma = inv_gamma / (1._prt + inv_gamma);

    // const amrex::ParticleReal gamma = 1.0 / inv_gamma;  // 计算gamma
    // ux = ux * gamma / PhysConst::c;
    // uy = uy * gamma / PhysConst::c;
    // uz = uz * gamma / PhysConst::c;

    // Scalar product of momentum and B field
    amrex::ParticleReal uB = (ux*Bx + uy*By + uz*Bz);

    // Define E/B field dependent parameters: tau and upsilon (ups).
    amrex::Real taux = tauconst * ((anom + inv_gamma) * Bx
                    - (anom + inv_1pgamma) * (uy * Ez - uz * Ey) * inv_gamma / PhysConst::c
                    - anom * inv_gamma * inv_1pgamma * uB * ux);
    amrex::Real tauy = tauconst * ((anom + inv_gamma) * By
                    - (anom + inv_1pgamma) * (uz * Ex - ux * Ez) * inv_gamma / PhysConst::c
                    - anom * inv_gamma * inv_1pgamma * uB * uy);
    amrex::Real tauz = tauconst * ((anom + inv_gamma) * Bz
                    - (anom + inv_1pgamma) * (ux * Ey - uy * Ex) * inv_gamma / PhysConst::c
                    - anom * inv_gamma * inv_1pgamma * uB * uz);

    amrex::Real tau2 = taux*taux + tauy*tauy + tauz*tauz;

    // ---------- Tamburini Method ---------------------------------------------
    amrex::Real ups = 1._prt / (1._prt + tau2);
    amrex::Real stau = sx_i * taux + sy_i * tauy + sz_i * tauz;

    // New spin components
    sx_f = ups * (sx_i + 2._prt * (sy_i * tauz - sz_i * tauy)
                + stau * taux + (sz_i * taux - sx_i * tauz) * tauz
                - (sx_i * tauy - sy_i * taux) * tauy);
    sy_f = ups * (sy_i + 2._prt * (sz_i * taux - sx_i * tauz)
                + stau * tauy + (sx_i * tauy - sy_i * taux) * taux
                - (sy_i * tauz - sz_i * tauy) * tauz);
    sz_f = ups * (sz_i + 2._prt * (sx_i * tauy - sy_i * taux)
                + stau * tauz + (sy_i * tauz - sz_i * tauy) * tauy
                - (sz_i * taux - sx_i * tauz) * taux);
    // -------------------------------------------------------------------------
}

#endif // WARPX_SPIN_PUSHER_UPDATESPIN_H_
