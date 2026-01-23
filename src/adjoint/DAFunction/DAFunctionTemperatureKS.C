/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionTemperatureKS.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionTemperatureKS, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionTemperatureKS, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionTemperatureKS::DAFunctionTemperatureKS(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName)
{

    functionDict_.readEntry<scalar>("coeffKS", coeffKS_);
    isGrad_ = functionDict_.lookupOrDefault<bool>("isGrad", false);
}

/// calculate the value of objective function
scalar DAFunctionTemperatureKS::calcFunction()
{
    /*
    Description:
        Calculate the maximal Temperature aggregated using the KS function
        where the KS function KS(x) = 1/coeffKS * ln( sum[exp(coeffKS*x_i)] ) 
    */

    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");

    volScalarField tmpField = T;
    if (isGrad_)
    {
        tmpField = mag(fvc::grad(T)) * dimensionedScalar("unitLength", dimLength, 1.0);
    }

    scalar maxValue = 0;
    scalar tmpMaxValue = gMax(tmpField);
    assignValueCheckAD(maxValue, tmpMaxValue);

    scalar objValTmp = 0.0;
    forAll(cellSources_, idxI)
    {
        const label& cellI = cellSources_[idxI];
        objValTmp += exp(coeffKS_ * scale_ * (tmpField[cellI] - maxValue));

        if (objValTmp > 1e200)
        {
            FatalErrorIn(" ") << "KS function summation term too large! "
                              << "Reduce coeffKS! " << abort(FatalError);
        }
    }

    // need to reduce the sum of force across all processors
    reduce(objValTmp, sumOp<scalar>());

    functionValue = log(objValTmp) / coeffKS_ + scale_ * maxValue;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
