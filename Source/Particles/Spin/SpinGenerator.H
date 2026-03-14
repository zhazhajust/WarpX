#ifndef SPINGENERATOR_H
#define SPINGENERATOR_H

#include "Utils/WarpXConst.H"

#include <memory.h>
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>
#include <cmath>

class SpinGenerator
{
public:
    SpinGenerator () = default;
    virtual ~SpinGenerator() = default;

    SpinGenerator(const SpinGenerator&) = default;
    SpinGenerator& operator=(const SpinGenerator&) = default;
    SpinGenerator(SpinGenerator&&) = default;
    SpinGenerator& operator=(SpinGenerator&&) = default;

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    virtual void operator() (amrex::Real& sx, amrex::Real& sy, amrex::Real& sz, 
        amrex::RandomEngine const& engine) const noexcept
    {
        amrex::ignore_unused(sx);
        amrex::ignore_unused(sy);
        amrex::ignore_unused(sz);
        amrex::ignore_unused(engine);
    }
};

class SpinGeneratorWrapper {
public:
    SpinGeneratorWrapper() = default;
    ~SpinGeneratorWrapper() {
        #ifdef AMREX_USE_GPU
            if (d_spin_ptr) {
                amrex::The_Arena()->free(d_spin_ptr);
            }
        #endif
    }

    SpinGeneratorWrapper(const SpinGeneratorWrapper&) = default;
    SpinGeneratorWrapper& operator=(const SpinGeneratorWrapper&) = default;
    SpinGeneratorWrapper(SpinGeneratorWrapper&&) = default;
    SpinGeneratorWrapper& operator=(SpinGeneratorWrapper&&) = default;

    void setSpinGenerator(std::unique_ptr<SpinGenerator> spin_ptr) {
        h_spin_ptr = std::move(spin_ptr);
        #ifdef AMREX_USE_GPU
            d_spin_ptr = static_cast<SpinGenerator*>(amrex::The_Arena()->alloc(sizeof(SpinGenerator)));
            amrex::Gpu::htod_memcpy_async(d_spin_ptr, h_spin_ptr.get(), sizeof(*h_spin_ptr));
        #else
            d_spin_ptr = h_spin_ptr.get();
        #endif
    }

    SpinGenerator* getSpinGenerator() {
        return d_spin_ptr;
    }

private:
    std::unique_ptr<SpinGenerator> h_spin_ptr = nullptr;
    SpinGenerator* d_spin_ptr = nullptr;
};

class RandSpinGenerator: public SpinGenerator
{
public:
    /**
     * Constructor does nothing because optical depth initialization
     * does not require control parameters or lookup tables.
     */
    RandSpinGenerator () = default;

    /**
     * () operator is just a thin wrapper around a very simple function to
     * generate the optical depth. It can be used on GPU.
     */
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    void operator() (amrex::Real& sx, amrex::Real& sy, amrex::Real& sz, 
        amrex::RandomEngine const& engine) const noexcept override
    {
        // namespace pxr_qs = picsar::multi_physics::phys::quantum_sync;

        // //A random number in [0,1) should be provided as an argument.
        // return pxr_qs::get_optical_depth(amrex::Random(engine));
        const amrex::Real radius = 1.0;
        amrex::Real rand = amrex::Random(engine);
        amrex::Real z = rand * 2 - 1;
        rand = amrex::Random(engine);
        amrex::Real phi = rand * 2 * M_PI;
        // Calculate sin(theta) based on z
        amrex::Real sin_theta = std::sin(std::acos(z / radius));
        // Calculate cartesian coordinates x and y
        amrex::Real x = sin_theta * std::cos(phi);
        amrex::Real y = sin_theta * std::sin(phi);

        // Assign the values to output references
        sx = x;
        sy = y;
        sz = z;
    }
};

class UniformSpinGenerator: public SpinGenerator
{
public:
    /**
     * Constructor does nothing because optical depth initialization
     * does not require control parameters or lookup tables.
     */
    // UniformSpinGenerator () = default;
    UniformSpinGenerator (amrex::Real sx, amrex::Real sy, amrex::Real sz)
        : sx_m(sx), sy_m(sy), sz_m(sz) {}

    /**
     * () operator is just a thin wrapper around a very simple function to
     * generate the optical depth. It can be used on GPU.
     */
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    void operator() (amrex::Real& sx, amrex::Real& sy, amrex::Real& sz, 
        amrex::RandomEngine const& engine) const noexcept override
    {
        // Assign the values to output references
        sx = sx_m;
        sy = sy_m;
        sz = sz_m;
        amrex::ignore_unused(engine);
    }
protected:
    amrex::Real sx_m = 0.0;
    amrex::Real sy_m = 0.0;
    amrex::Real sz_m = 1.0;
};

// class RandSpinGenerator
// {
// public:
//     RandSpinGenerator (amrex::ParticleReal s_m, amrex::ParticleReal sx, amrex::ParticleReal sy, amrex::ParticleReal sz)
//         : s_m(s_m), alpha(0.0), target_sx(sx), target_sy(sy), target_sz(sz)
//     {
//         // Normalize the target direction vector
//         amrex::ParticleReal norm = std::sqrt(sx * sx + sy * sy + sz * sz);
//         target_sx /= norm;
//         target_sy /= norm;
//         target_sz /= norm;

//         if (s_m != 0.0 && std::abs(s_m) != 1.0) {
//             // Solve for alpha if s_m is not 0 or ±1
//             auto func = [this](amrex::ParticleReal alpha) -> amrex::ParticleReal {
//                 return 1.0 / alpha - 1.0 / std::tanh(alpha) - this->s_m;
//             };
//             alpha = brent(func, 0.1, 10.0, 1e-6); // Use a numerical solver like Brent's method
//         }
//     }

//     AMREX_GPU_HOST_DEVICE
//     AMREX_FORCE_INLINE
//     void operator() (amrex::ParticleReal& sx_out, amrex::ParticleReal& sy_out, amrex::ParticleReal& sz_out,
//                     amrex::RandomEngine const& engine) const noexcept
//     {
//         amrex::Real z;

//         if (std::abs(s_m) == 1.0) {
//             z = s_m;
//         } else if (s_m == 0.0) {
//             z = amrex::Random(engine) * 2.0 - 1.0; // random uniform between -1 and 1
//         } else {
//             // Generate z with probability distribution e^(-alpha z)
//             amrex::Real u = amrex::Random(engine);
//             z = -1.0 / alpha * std::log(std::exp(alpha) * (1.0 - u) + std::exp(-alpha) * u);
//         }

//         amrex::Real phi = amrex::Random(engine) * 2.0 * M_PI;
//         amrex::Real sin_theta = std::sqrt(1.0 - z * z);
//         amrex::Real x = sin_theta * std::cos(phi);
//         amrex::Real y = sin_theta * std::sin(phi);

//         // Rotate (x, y, z) to (sx_out, sy_out, sz_out)
//         rotate_to_target(x, y, z, sx_out, sy_out, sz_out);
//     }

// private:
//     amrex::ParticleReal s_m;
//     amrex::ParticleReal alpha;
//     amrex::ParticleReal target_sx, target_sy, target_sz;

//     // Implement a simple Brent's method for root finding
//     static amrex::ParticleReal brent(std::function<Real(Real)> func, amrex::Real lower, amrex::Real upper, amrex::Real tol) {
//         amrex::ParticleReal a = lower;
//         amrex::ParticleReal b = upper;
//         amrex::ParticleReal c = a;
//         amrex::ParticleReal fa = func(a);
//         amrex::ParticleReal fb = func(b);
//         amrex::ParticleReal fc = fa;

//         for (int iter = 0; iter < 100; ++iter) {
//             if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
//                 c = a;
//                 fc = fa;
//                 amrex::ParticleReal d = b - a;
//                 amrex::ParticleReal e = d;
//             }

//             if (std::abs(fc) < std::abs(fb)) {
//                 a = b;
//                 b = c;
//                 c = a;
//                 fa = fb;
//                 fb = fc;
//                 fc = fa;
//             }

//             amrex::ParticleReal tol1 = 2.0 * std::numeric_limits<amrex::ParticleReal>::epsilon() * std::abs(b) + 0.5 * tol;
//             amrex::ParticleReal xm = 0.5 * (c - b);

//             if (std::abs(xm) <= tol1 || fb == 0.0) {
//                 return b;
//             }

//             amrex::ParticleReal p, q, r;
//             if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
//                 amrex::ParticleReal s = fb / fa;
//                 if (a == c) {
//                     p = 2.0 * xm * s;
//                     q = 1.0 - s;
//                 } else {
//                     q = fa / fc;
//                     r = fb / fc;
//                     p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
//                     q = (q - 1.0) * (r - 1.0) * (s - 1.0);
//                 }

//                 if (p > 0.0) q = -q;
//                 p = std::abs(p);
//                 amrex::ParticleReal min1 = 3.0 * xm * q - std::abs(tol1 * q);
//                 amrex::ParticleReal min2 = std::abs(e * q);
//                 if (2.0 * p < (min1 < min2 ? min1 : min2)) {
//                     e = d;
//                     d = p / q;
//                 } else {
//                     d = xm;
//                     e = d;
//                 }
//             } else {
//                 d = xm;
//                 e = d;
//             }

//             amrex::ParticleReal d = xm;
//             a = b;
//             fa = fb;
//             if (std::abs(d) > tol1) {
//                 b += d;
//             } else {
//                 b += (xm > 0 ? tol1 : -tol1);
//             }
//             fb = func(b);
//         }

//         return b;
//     }

//     AMREX_GPU_HOST_DEVICE
//     AMREX_FORCE_INLINE
//     void rotate_to_target(amrex::ParticleReal x, amrex::ParticleReal y, amrex::ParticleReal z, amrex::ParticleReal& sx_out, amrex::ParticleReal& sy_out, amrex::ParticleReal& sz_out) const {
//         // Calculate rotation axis (cross product of (0,0,1) and target direction)
//         amrex::ParticleReal Kx = -target_sy;
//         amrex::ParticleReal Ky = target_sx;
//         amrex::ParticleReal Kz = 0.0;
//         amrex::ParticleReal K_norm = std::sqrt(Kx * Kx + Ky * Ky + Kz * Kz);

//         if (K_norm > 0.0) {
//             Kx /= K_norm;
//             Ky /= K_norm;
//             Kz /= K_norm;
//         }

//         amrex::ParticleReal cos_theta = target_sz;
//         amrex::ParticleReal sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

//         amrex::ParticleReal dot = x * Kx + y * Ky + z * Kz;
//         amrex::ParticleReal cross_x = Ky * z - Kz * y;
//         amrex::ParticleReal cross_y = Kz * x - Kx * z;
//         amrex::ParticleReal cross_z = Kx * y - Ky * x;

//         sx_out = cos_theta * x + sin_theta * cross_x + dot * (1.0 - cos_theta) * Kx;
//         sy_out = cos_theta * y + sin_theta * cross_y + dot * (1.0 - cos_theta) * Ky;
//         sz_out = cos_theta * z + sin_theta * cross_z + dot * (1.0 - cos_theta) * Kz;
//     }
// };

#endif // SPINGENERATOR_H
