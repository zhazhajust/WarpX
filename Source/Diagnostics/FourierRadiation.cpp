#include "FourierRadiation.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include "ablastr/warn_manager/WarnManager.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace amrex;
namespace fr = warpx::diagnostics::fourier_radiation;

namespace
{
    static constexpr int n_record_reals = 10;

    enum FlatRecordComponent : int
    {
        record_x_mid = 0,
        record_y_mid,
        record_z_mid,
        record_ux_old,
        record_uy_old,
        record_uz_old,
        record_ux_new,
        record_uy_new,
        record_uz_new,
        record_weight
    };

    void PackParticleRecord (
        FourierRadiationParticleRecord const& record,
        amrex::Real* flat_record);

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    FourierRadiationParticleRecord UnpackParticleRecord (
        amrex::Real const* flat_record) noexcept;

    amrex::Vector<amrex::Real> FlattenParticleRecords (
        FourierRadiationParticleRecord const* records,
        amrex::Long num_records);

    void CopyFlatRecordRange (
        amrex::Real const* src_records,
        amrex::Long src_begin,
        amrex::Real* dst_records,
        amrex::Long dst_begin,
        amrex::Long num_records);

    amrex::Vector<amrex::Real> RedistributeParticleRecords (
        FourierRadiationParticleRecord const* local_records,
        amrex::Long local_count,
        amrex::Vector<amrex::Long> const& counts);

    void AccumulateDeviceParticleRecords (
        FourierRadiationDeviceData radiation_data,
        amrex::Real const* flat_records,
        amrex::Long num_records,
        amrex::Real dt,
        amrex::Real charge,
        amrex::Real radiation_time);
}

FourierRadiation::FourierRadiation ()
{
    ParmParse const pp_warpx("warpx");
    Vector<std::string> rd_names;
    pp_warpx.queryarr("reduced_diags_names", rd_names);

    std::string rd_name;
    int num_fourier_radiation_diags = 0;
    for (auto const& name : rd_names) {
        ParmParse const pp_rd(name);
        std::string rd_type;
        if (pp_rd.query("type", rd_type) && rd_type == "FourierRadiation") {
            rd_name = name;
            ++num_fourier_radiation_diags;
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_fourier_radiation_diags <= 1,
        "Only one FourierRadiation reduced diagnostic is currently supported.");

    if (num_fourier_radiation_diags == 0) {
        return;
    }
    m_enabled = true;

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("Inline Fourier radiation is currently implemented only for 3D and RZ.");
#endif

    ParmParse const pp_fr(rd_name);
    pp_fr.get("species", m_species_name);

    Vector<Real> frequency_params;
    Vector<Real> theta_params;
    Vector<Real> phi_params;
    utils::parser::getArrWithParser(pp_fr, "frequencies", frequency_params);
    utils::parser::getArrWithParser(pp_fr, "theta", theta_params);
    utils::parser::getArrWithParser(pp_fr, "phi", phi_params);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        frequency_params.size() == 3 && theta_params.size() == 3 && phi_params.size() == 3,
        rd_name + ".frequencies/theta/phi must each be: min max num_points.");

    m_omega_min = frequency_params[0];
    m_omega_max = frequency_params[1];
    m_num_omega = static_cast<int>(std::llround(frequency_params[2]));
    m_theta_min = theta_params[0];
    m_theta_max = theta_params[1];
    m_num_theta = static_cast<int>(std::llround(theta_params[2]));
    m_phi_min = phi_params[0];
    m_phi_max = phi_params[1];
    m_num_phi = static_cast<int>(std::llround(phi_params[2]));
    pp_fr.query("frequency_grid", m_omega_grid);
    std::string work_distribution = "balanced";
    pp_fr.query("work_distribution", work_distribution);
    utils::parser::queryWithParser(
        pp_fr,
        "work_distribution_imbalance_threshold",
        m_work_distribution_imbalance_threshold);
    pp_fr.query("work_distribution_stats", m_work_distribution_stats);
    if (work_distribution == "local") {
        m_work_distribution = WorkDistribution::Local;
    } else if (work_distribution == "balanced") {
        m_work_distribution = WorkDistribution::Balanced;
    } else if (work_distribution == "auto") {
        m_work_distribution = WorkDistribution::Auto;
    }
    utils::parser::queryWithParser(pp_fr, "particle_fraction", m_particle_fraction);
    std::string particle_filter_mode = "step";
    pp_fr.query("filter_mode", particle_filter_mode);
    m_particle_filter_is_latched = particle_filter_mode == "latched";
    std::string particle_filter_string;
    m_do_particle_filter = pp_fr.query(
        "filter_function(t,x,y,z,ux,uy,uz,w)", particle_filter_string);
    if (m_do_particle_filter) {
        utils::parser::Store_parserString(
            pp_fr,
            "filter_function(t,x,y,z,ux,uy,uz,w)",
            particle_filter_string);
        m_particle_filter_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(
                particle_filter_string, {"t", "x", "y", "z", "ux", "uy", "uz", "w"}));
        m_particle_filter_function = m_particle_filter_parser->compile<8>();
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_num_omega > 0 && m_num_theta > 0 && m_num_phi > 0,
        "Fourier radiation grid dimensions must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_omega_min > 0._rt && m_omega_max > 0._rt,
        rd_name + ".frequencies bounds must be positive.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_omega_grid == "linear" || m_omega_grid == "log",
        rd_name + ".frequency_grid must be either 'linear' or 'log'.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        particle_filter_mode == "step" || particle_filter_mode == "latched",
        rd_name + ".filter_mode must be either 'step' or 'latched'.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_particle_filter_is_latched || m_do_particle_filter,
        rd_name + ".filter_mode = latched requires a filter_function.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_particle_fraction > 0._rt && m_particle_fraction <= 1._rt,
        rd_name + ".particle_fraction must be in the interval (0, 1].");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        work_distribution == "local" || work_distribution == "balanced"
            || work_distribution == "auto",
        rd_name + ".work_distribution must be one of: local, balanced, auto.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_work_distribution_imbalance_threshold >= 1._rt,
        rd_name + ".work_distribution_imbalance_threshold must be >= 1.");

    pp_fr.query("reset_after_output", m_reset_after_output);

    Allocate();
}

bool
FourierRadiation::UseImmediateLocalAccumulation () const
{
    return m_work_distribution == WorkDistribution::Local;
}

bool
FourierRadiation::IsSpeciesSelected (std::string const& species_name) const
{
    return m_enabled && species_name == m_species_name;
}

void
FourierRadiation::Allocate ()
{
    m_num_grid_nodes = m_num_omega * m_num_theta * m_num_phi;

    Vector<Real> h_grid_data(m_num_omega + 2*m_num_theta + 2*m_num_phi);
    Real* const h_omega = h_grid_data.data();
    Real* const h_sin_theta = h_omega + m_num_omega;
    Real* const h_cos_theta = h_sin_theta + m_num_theta;
    Real* const h_sin_phi = h_cos_theta + m_num_theta;
    Real* const h_cos_phi = h_sin_phi + m_num_phi;
    m_frequency.resize(m_num_omega);

    for (int i = 0; i < m_num_omega; ++i) {
        Real const frac = (m_num_omega == 1) ? 0._rt : Real(i) / Real(m_num_omega - 1);
        if (m_omega_grid == "log") {
            m_frequency[i] = std::exp(
                std::log(m_omega_min) + frac * (std::log(m_omega_max) - std::log(m_omega_min)));
        } else {
            m_frequency[i] = m_omega_min + frac * (m_omega_max - m_omega_min);
        }
        h_omega[i] = 2._rt * Math::pi<Real>() * m_frequency[i];
    }
    for (int i = 0; i < m_num_theta; ++i) {
        Real const frac = (m_num_theta == 1) ? 0._rt : Real(i) / Real(m_num_theta - 1);
        Real const theta = m_theta_min + frac * (m_theta_max - m_theta_min);
        h_sin_theta[i] = std::sin(theta);
        h_cos_theta[i] = std::cos(theta);
    }
    for (int i = 0; i < m_num_phi; ++i) {
        Real const phi = m_phi_min + (m_phi_max - m_phi_min) / Real(m_num_phi) * Real(i);
        h_sin_phi[i] = std::sin(phi);
        h_cos_phi[i] = std::cos(phi);
    }

    m_grid_data.resize(h_grid_data.size());
    Gpu::copy(Gpu::hostToDevice, h_grid_data.begin(), h_grid_data.end(), m_grid_data.begin());

    m_num_amp_streams = Gpu::numGpuStreams();
    m_amp.resize(
        static_cast<Long>(m_num_amp_streams) * fr::n_amp_components * m_num_grid_nodes, 0._rt);
}

FourierRadiationDeviceData
FourierRadiation::GetDeviceData ()
{
    Long stream_offset = 0;
#ifdef AMREX_USE_GPU
    stream_offset = static_cast<Long>(Gpu::Device::streamIndex())
        * fr::n_amp_components * m_num_grid_nodes;
#endif

    return FourierRadiationDeviceData{
        OmegaData(),
        SinThetaData(),
        CosThetaData(),
        SinPhiData(),
        CosPhiData(),
        m_amp.dataPtr() + stream_offset,
        m_num_omega,
        m_num_theta,
        m_num_phi,
        m_num_grid_nodes,
        m_particle_fraction};
}

void
FourierRadiation::GetIntensity (std::vector<Real>& intensity) const
{
    intensity.assign(m_num_grid_nodes, 0._rt);
    if (!m_enabled) {
        return;
    }

    Vector<Real> amp(
        static_cast<Long>(m_num_amp_streams) * fr::n_amp_components * m_num_grid_nodes);
    Gpu::copy(Gpu::deviceToHost, m_amp.begin(), m_amp.end(), amp.begin());
    ParallelDescriptor::ReduceRealSum(amp.data(), static_cast<int>(amp.size()));

    for (int i = 0; i < m_num_grid_nodes; ++i) {
        constexpr Real prefactor =
            1._rt / (16._rt * Math::pi<Real>() * Math::pi<Real>() * Math::pi<Real>()
                     * PhysConst::epsilon_0 * PhysConst::c);
        Real axr = 0._rt;
        Real axi = 0._rt;
        Real ayr = 0._rt;
        Real ayi = 0._rt;
        Real azr = 0._rt;
        Real azi = 0._rt;
        for (int istream = 0; istream < m_num_amp_streams; ++istream) {
            Real const* const stream_amp = amp.data()
                + static_cast<Long>(istream) * fr::n_amp_components * m_num_grid_nodes;
            axr += stream_amp[fr::amp_x_re * m_num_grid_nodes + i];
            axi += stream_amp[fr::amp_x_im * m_num_grid_nodes + i];
            ayr += stream_amp[fr::amp_y_re * m_num_grid_nodes + i];
            ayi += stream_amp[fr::amp_y_im * m_num_grid_nodes + i];
            azr += stream_amp[fr::amp_z_re * m_num_grid_nodes + i];
            azi += stream_amp[fr::amp_z_im * m_num_grid_nodes + i];
        }
        intensity[i] = prefactor * (axr*axr + axi*axi
                                  + ayr*ayr + ayi*ayi
                                  + azr*azr + azi*azi);
    }
}

void
FourierRadiation::FlushParticleRecords (
    FourierRadiationParticleRecord const* local_records,
    Long const num_local_records,
    Real const dt,
    Real const charge,
    Real const radiation_time)
{
    if (!m_enabled || m_work_distribution == WorkDistribution::Local) {
        return;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        num_local_records >= 0,
        "FourierRadiation local record count must be non-negative.");

    int const nprocs = ParallelDescriptor::NProcs();
    Long const local_count = num_local_records;
    Vector<Long> counts(nprocs, 0);
    ParallelAllGather::AllGather(
        local_count, counts.data(), ParallelDescriptor::Communicator());

    Long global_count = 0;
    Long max_count = 0;
    Long min_count = counts.empty() ? 0 : counts[0];
    for (Long const count : counts) {
        global_count += count;
        max_count = std::max(max_count, count);
        min_count = std::min(min_count, count);
    }

    Real const avg_count =
        (nprocs > 0) ? static_cast<Real>(global_count) / static_cast<Real>(nprocs) : 0._rt;
    Real const imbalance =
        (avg_count > 0._rt) ? static_cast<Real>(max_count) / avg_count : 0._rt;
    bool const use_balanced =
        m_work_distribution == WorkDistribution::Balanced
        || (m_work_distribution == WorkDistribution::Auto
            && imbalance >= m_work_distribution_imbalance_threshold);

    if (m_work_distribution_stats && ParallelDescriptor::IOProcessor()) {
        std::string const chosen_mode = use_balanced ? "balanced" : "local";
        Long const estimated_record_bytes =
            global_count * static_cast<Long>(n_record_reals * sizeof(Real));
        ablastr::warn_manager::WMRecordWarning(
            "FourierRadiation",
            "work_distribution selected_count min/avg/max = "
                + std::to_string(min_count) + "/"
                + std::to_string(static_cast<double>(avg_count)) + "/"
                + std::to_string(max_count)
                + ", imbalance = " + std::to_string(static_cast<double>(imbalance))
                + ", estimated record bytes = " + std::to_string(estimated_record_bytes)
                + ", chosen mode = " + chosen_mode);
    }

    if (global_count == 0) {
        return;
    }

    Vector<Real> records_for_compute = use_balanced
        ? RedistributeParticleRecords(local_records, local_count, counts)
        : FlattenParticleRecords(local_records, local_count);

    Long const num_records_to_compute =
        static_cast<Long>(records_for_compute.size()) / n_record_reals;
    if (num_records_to_compute <= 0) {
        return;
    }

    Gpu::AsyncVector<Real> device_records(records_for_compute.size());
    Gpu::copy(
        Gpu::hostToDevice,
        records_for_compute.begin(),
        records_for_compute.end(),
        device_records.begin());

    AccumulateDeviceParticleRecords(
        GetDeviceData(), device_records.dataPtr(), num_records_to_compute, dt, charge,
        radiation_time);
}

namespace
{

void
PackParticleRecord (
    FourierRadiationParticleRecord const& record,
    Real* const flat_record)
{
    flat_record[record_x_mid] = record.x_mid;
    flat_record[record_y_mid] = record.y_mid;
    flat_record[record_z_mid] = record.z_mid;
    flat_record[record_ux_old] = record.ux_old;
    flat_record[record_uy_old] = record.uy_old;
    flat_record[record_uz_old] = record.uz_old;
    flat_record[record_ux_new] = record.ux_new;
    flat_record[record_uy_new] = record.uy_new;
    flat_record[record_uz_new] = record.uz_new;
    flat_record[record_weight] = record.weight;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FourierRadiationParticleRecord
UnpackParticleRecord (Real const* const flat_record) noexcept
{
    return FourierRadiationParticleRecord{
        flat_record[record_x_mid],
        flat_record[record_y_mid],
        flat_record[record_z_mid],
        flat_record[record_ux_old],
        flat_record[record_uy_old],
        flat_record[record_uz_old],
        flat_record[record_ux_new],
        flat_record[record_uy_new],
        flat_record[record_uz_new],
        flat_record[record_weight]};
}

Vector<Real>
FlattenParticleRecords (
    FourierRadiationParticleRecord const* const records,
    Long const num_records)
{
    Vector<Real> flat_records(static_cast<std::size_t>(num_records) * n_record_reals);
    for (Long i = 0; i < num_records; ++i) {
        PackParticleRecord(records[i], flat_records.data() + i * n_record_reals);
    }
    return flat_records;
}

void
CopyFlatRecordRange (
    Real const* const src_records,
    Long const src_begin,
    Real* const dst_records,
    Long const dst_begin,
    Long const num_records)
{
    std::copy_n(
        src_records + src_begin * n_record_reals,
        static_cast<std::size_t>(num_records) * n_record_reals,
        dst_records + dst_begin * n_record_reals);
}

Vector<Real>
RedistributeParticleRecords (
    FourierRadiationParticleRecord const* const local_records,
    Long const local_count,
    Vector<Long> const& counts)
{
    int const nprocs = ParallelDescriptor::NProcs();
    int const myproc = ParallelDescriptor::MyProc();

    Vector<Long> source_begin(nprocs + 1, 0);
    for (int i = 0; i < nprocs; ++i) {
        source_begin[i + 1] = source_begin[i] + counts[i];
    }
    Long const global_count = source_begin[nprocs];
    Long const my_source_begin = source_begin[myproc];
    Long const my_source_end = source_begin[myproc + 1];
    Long const my_target_begin = global_count * myproc / nprocs;
    Long const my_target_end = global_count * (myproc + 1) / nprocs;
    Long const my_target_count = my_target_end - my_target_begin;

    Vector<Real> local_flat_records = FlattenParticleRecords(local_records, local_count);
    Vector<Real> redistributed_records(
        static_cast<std::size_t>(my_target_count) * n_record_reals);

    Vector<ParallelDescriptor::Message> recv_messages;
    Vector<ParallelDescriptor::Message> send_messages;
    Vector<Vector<Real>> send_buffers(nprocs);

    int const tag = ParallelDescriptor::SeqNum();

    for (int src = 0; src < nprocs; ++src) {
        if (src == myproc) {
            continue;
        }
        Long const overlap_begin = std::max(my_target_begin, source_begin[src]);
        Long const overlap_end = std::min(my_target_end, source_begin[src + 1]);
        Long const num_records = overlap_end - overlap_begin;
        if (num_records <= 0) {
            continue;
        }
        Long const dst_offset = overlap_begin - my_target_begin;
        Real* const recv_ptr = redistributed_records.data() + dst_offset * n_record_reals;
        Long const num_reals = num_records * n_record_reals;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            num_reals <= static_cast<Long>(std::numeric_limits<int>::max()),
            "FourierRadiation redistributed receive buffer is too large for AMReX communication.");
        recv_messages.push_back(
            ParallelDescriptor::Arecv(recv_ptr, static_cast<std::size_t>(num_reals), src, tag));
    }

    for (int dst = 0; dst < nprocs; ++dst) {
        Long const dst_begin = global_count * dst / nprocs;
        Long const dst_end = global_count * (dst + 1) / nprocs;
        Long const overlap_begin = std::max(my_source_begin, dst_begin);
        Long const overlap_end = std::min(my_source_end, dst_end);
        Long const num_records = overlap_end - overlap_begin;
        if (num_records <= 0) {
            continue;
        }

        Long const src_offset = overlap_begin - my_source_begin;
        if (dst == myproc) {
            Long const dst_offset = overlap_begin - my_target_begin;
            CopyFlatRecordRange(
                local_flat_records.data(),
                src_offset,
                redistributed_records.data(),
                dst_offset,
                num_records);
            continue;
        }

        Long const num_reals = num_records * n_record_reals;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            num_reals <= static_cast<Long>(std::numeric_limits<int>::max()),
            "FourierRadiation redistributed send buffer is too large for AMReX communication.");
        Vector<Real>& send_buffer = send_buffers[dst];
        send_buffer.resize(static_cast<std::size_t>(num_reals));
        CopyFlatRecordRange(
            local_flat_records.data(),
            src_offset,
            send_buffer.data(),
            0,
            num_records);
        send_messages.push_back(
            ParallelDescriptor::Asend(send_buffer.data(), send_buffer.size(), dst, tag));
    }

    for (auto& message : recv_messages) {
        message.wait();
    }
    for (auto& message : send_messages) {
        message.wait();
    }

    return redistributed_records;
}

void
AccumulateDeviceParticleRecords (
    FourierRadiationDeviceData const radiation_data,
    Real const* flat_records,
    Long const num_records,
    Real const dt,
    Real const charge,
    Real const radiation_time)
{
    if (num_records <= 0) {
        return;
    }

    constexpr long particles_per_radiation_chunk = 128;
    constexpr std::size_t radiation_partial_workspace_bytes = 64u * 1024u * 1024u;

    const long num_chunks =
        (num_records + particles_per_radiation_chunk - 1) / particles_per_radiation_chunk;
    const auto bytes_per_chunk = static_cast<std::size_t>(fr::n_amp_components)
        * static_cast<std::size_t>(radiation_data.num_grid_nodes) * sizeof(Real);
    const long chunks_per_batch = std::max(
        1L,
        static_cast<long>(radiation_partial_workspace_bytes / std::max<std::size_t>(
            bytes_per_chunk, 1u)));

    for (long chunk_begin = 0; chunk_begin < num_chunks; chunk_begin += chunks_per_batch) {
        const long batch_chunks = std::min(chunks_per_batch, num_chunks - chunk_begin);
        const Long partial_size = static_cast<Long>(fr::n_amp_components)
            * static_cast<Long>(radiation_data.num_grid_nodes) * batch_chunks;
        Gpu::AsyncVector<Real> partial_amp(partial_size);
        Real* const AMREX_RESTRICT partial_amp_ptr = partial_amp.dataPtr();

        const Long partial_work_size =
            static_cast<Long>(batch_chunks) * radiation_data.num_grid_nodes;
        amrex::ParallelFor(partial_work_size, [=] AMREX_GPU_DEVICE (Long local_iwork)
        {
            const long local_chunk = static_cast<long>(
                local_iwork / radiation_data.num_grid_nodes);
            const int gti = static_cast<int>(
                local_iwork - local_chunk * radiation_data.num_grid_nodes);
            const long chunk = chunk_begin + local_chunk;
            const Long particle_begin =
                static_cast<Long>(chunk) * particles_per_radiation_chunk;
            const Long particle_end =
                (particle_begin + particles_per_radiation_chunk < num_records)
                ? particle_begin + particles_per_radiation_chunk : num_records;

            FourierRadiationAmplitude sum;
            for (Long local_record_i = particle_begin;
                 local_record_i < particle_end;
                 ++local_record_i)
            {
                FourierRadiationParticleRecord const record =
                    UnpackParticleRecord(flat_records + local_record_i * n_record_reals);
                FourierRadiationAmplitude const contribution =
                    fr::contribution(radiation_data, record, gti, dt, charge, radiation_time);
                sum.x_re += contribution.x_re;
                sum.x_im += contribution.x_im;
                sum.y_re += contribution.y_re;
                sum.y_im += contribution.y_im;
                sum.z_re += contribution.z_re;
                sum.z_im += contribution.z_im;
            }

            const Long offset_base =
                (static_cast<Long>(local_chunk) * fr::n_amp_components
                 * radiation_data.num_grid_nodes) + gti;
            partial_amp_ptr[
                offset_base + fr::amp_x_re * radiation_data.num_grid_nodes] = sum.x_re;
            partial_amp_ptr[
                offset_base + fr::amp_x_im * radiation_data.num_grid_nodes] = sum.x_im;
            partial_amp_ptr[
                offset_base + fr::amp_y_re * radiation_data.num_grid_nodes] = sum.y_re;
            partial_amp_ptr[
                offset_base + fr::amp_y_im * radiation_data.num_grid_nodes] = sum.y_im;
            partial_amp_ptr[
                offset_base + fr::amp_z_re * radiation_data.num_grid_nodes] = sum.z_re;
            partial_amp_ptr[
                offset_base + fr::amp_z_im * radiation_data.num_grid_nodes] = sum.z_im;
        });

        const Long reduce_work_size =
            static_cast<Long>(fr::n_amp_components) * radiation_data.num_grid_nodes;
        amrex::ParallelFor(reduce_work_size, [=] AMREX_GPU_DEVICE (Long iwork)
        {
            const int component = static_cast<int>(iwork / radiation_data.num_grid_nodes);
            const int gti = static_cast<int>(
                iwork - static_cast<Long>(component) * radiation_data.num_grid_nodes);

            Real sum = 0._rt;
            for (long local_chunk = 0; local_chunk < batch_chunks; ++local_chunk) {
                const Long partial_i =
                    static_cast<Long>(local_chunk) * fr::n_amp_components
                    * radiation_data.num_grid_nodes
                    + static_cast<Long>(component) * radiation_data.num_grid_nodes
                    + gti;
                sum += partial_amp_ptr[partial_i];
            }
            radiation_data.amp[iwork] += sum;
        });
    }
}

}

void
FourierRadiation::Reset ()
{
    if (!m_enabled) {
        return;
    }

    Real* const amp = m_amp.dataPtr();
    int const n = static_cast<int>(m_amp.size());

    ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
    {
        amp[i] = 0._rt;
    });
}
