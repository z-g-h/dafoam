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
#include "pointFields.H"

namespace Foam
{

class DAOutputPointTemperature : public DAOutput
{
protected:
    label size_;
    word fieldName_;
    scalar TRef_;

    static void unpackAndGlobalSum(
        const scalarList& rawOutput,
        labelList& pointToGlobal,
        List<double>& globalNumerator,
        List<double>& globalDenominator);

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

    virtual ~DAOutputPointTemperature()
    {
    }

    virtual void run(scalarList& output);

    virtual label size()
    {
        return size_;
    }

    virtual label distributed()
    {
        return 1;
    }

    // Reverse-AD interface. The physical output still has size_ entries, while
    // the local tape records [N_local, D_local] and stores passive global IDs.
    virtual label jacobianOutputSize();

    virtual label jacobianRegisteredOutputSize();

    virtual void runForJacobian(scalarList& output);

    virtual void setJacobianOutputSeeds(
        scalarList& output,
        const double* seed);

    // Rank-local [N, D, globalPointID] used as CoDiPack tape outputs.
    // This method contains no active-scalar MPI communication.
    void runLocalRaw(scalarList& rawOutput);

    label localRawSize() const
    {
        return 3 * size_;
    }

    // Assemble the physical point temperature from rank-local raw data.
    static void assemblePointTemperature(
        const scalarList& rawOutput,
        const scalar TRef,
        scalarList& pointTemperature);

    // Apply the exact transpose of T = sum(N)/sum(D) - TRef.
    static void calcReverseSeeds(
        const scalarList& rawOutput,
        const double* pointTemperatureSeed,
        List<double>& numeratorSeed,
        List<double>& denominatorSeed);
};

} // End namespace Foam

#endif
