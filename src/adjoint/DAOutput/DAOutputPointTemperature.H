/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Output the temperature at every local OpenFOAM mesh point. This is intended
    for one-way transfer from the converged CHT solid domain to a matching TACS
    thermoelastic volume mesh.

\*---------------------------------------------------------------------------*/

#ifndef DAOutputPointTemperature_H
#define DAOutputPointTemperature_H

#include "DAOutput.H"
#include "addToRunTimeSelectionTable.H"
#include "volPointInterpolation.H"
#include "pointFields.H"

namespace Foam
{

class DAOutputPointTemperature : public DAOutput
{
protected:
    label size_;
    word fieldName_;
    scalar TRef_;

public:
    TypeName("pointTemperatureOutput");

    DAOutputPointTemperature(
        const word outputName,
        const word outputType,
        fvMesh& mesh,
        const DAOption& daOption,
        DAModel& daModel,
        const DAIndex& daIndex,
        DAResidual& daResidual,
        UPtrList<DAFunction>& daFunctionList);

    virtual ~DAOutputPointTemperature() {}

    virtual void run(scalarList& output);

    virtual label size() { return size_; }

    virtual label distributed() { return 1; }
};

} // End namespace Foam

#endif
