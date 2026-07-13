/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAOutputPointTemperature.H"

namespace Foam
{

defineTypeNameAndDebug(DAOutputPointTemperature, 0);
addToRunTimeSelectionTable(DAOutput, DAOutputPointTemperature, dictionary);

DAOutputPointTemperature::DAOutputPointTemperature(
    const word outputName,
    const word outputType,
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
    : DAOutput(
        outputName,
        outputType,
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        daFunctionList)
{
    const dictionary& outputDict =
        daOption_.getAllOptions().subDict("outputInfo").subDict(outputName_);
    fieldName_ = outputDict.lookupOrDefault<word>("field", "T");
    TRef_ = outputDict.lookupOrDefault<scalar>("TRef", 0.0);
    size_ = mesh_.nPoints();
}

void DAOutputPointTemperature::run(scalarList& output)
{
    if (output.size() != size_)
    {
        FatalErrorInFunction
            << "Output size " << output.size()
            << " does not match mesh.nPoints() = " << size_
            << exit(FatalError);
    }

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>(fieldName_);

    tmp<pointScalarField> tPointT =
        volPointInterpolation::New(mesh_).interpolate(T);
    const pointScalarField& pointT = tPointT();

    forAll(pointT, pointI)
    {
        output[pointI] = pointT[pointI] - TRef_;
    }
}

} // End namespace Foam
