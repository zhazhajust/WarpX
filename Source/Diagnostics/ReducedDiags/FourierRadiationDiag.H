#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_FOURIERRADIATIONDIAG_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_FOURIERRADIATIONDIAG_H_

#include "ReducedDiags.H"

class FourierRadiationDiag
    : public ReducedDiags
{
public:
    explicit FourierRadiationDiag (std::string const& rd_name);

    void ComputeDiags (int step) override;
    void WriteToFile (int step) const override;
};

#endif
