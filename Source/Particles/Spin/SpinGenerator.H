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

#endif // SPINGENERATOR_H