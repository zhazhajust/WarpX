/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_TIMEDOMAINRADIATIONDIAG_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_TIMEDOMAINRADIATIONDIAG_H_

#include "ReducedDiags.H"

class TimeDomainRadiationDiag : public ReducedDiags
{
public:
    explicit TimeDomainRadiationDiag (std::string const& rd_name);

    void ComputeDiags (int step) override;
    void WriteToFile (int step) const override;
};

#endif
