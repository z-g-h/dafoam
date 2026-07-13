/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionScalarFieldKS.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionScalarFieldKS, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionScalarFieldKS, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionScalarFieldKS::DAFunctionScalarFieldKS(
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

    functionDict_.readEntry<word>("fieldName", fieldName_);

    functionDict_.readEntry<scalar>("coeffKS", coeffKS_);

    isGrad_ = functionDict_.lookupOrDefault<bool>("isGrad", false);

    normalMode_ = functionDict_.lookupOrDefault<word>("normalMode", "auto");
    if (normalMode_ == "fixed")
    {
        functionDict_.readEntry<scalar>("normalValue", normalValue_);
    }

    ksAggregationType_ = functionDict_.lookupOrDefault<word>("ksAggregationType", "continuous");

    alpha_ = functionDict_.lookupOrDefault<scalar>("alpha", 1.0);
}

/// calculate the value of objective function
scalar DAFunctionScalarFieldKS::calcFunction()
{
    /*
    Description:
        Calculate the maximal field aggregated using the KS function
        where the KS function KS(x) = 1/coeffKS * ln( sum[w * exp(coeffKS*x_i)] ) 
        w is a volume weights which is wi = Vi / sum(Vi)
    */

    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& field = db.lookupObject<volScalarField>(fieldName_);

    volScalarField tmpField = field;
    if (isGrad_)
    {
        tmpField = mag(fvc::grad(field)) * dimensionedScalar("unitLength", dimLength, 1.0);
    }

    scalar maxValue = 0;
    if (normalMode_ == "auto")
    {
        scalar tmpMaxValue = gMax(tmpField);
        assignValueCheckAD(maxValue, tmpMaxValue);
    }
    else if (normalMode_ == "fixed")
    {
        assignValueCheckAD(maxValue, normalValue_);
    }
    else
    {
        FatalErrorIn(" ") << "KS function normalMode must be auto or fixed! " << abort(FatalError);
    }

    scalar objValTmp = 0.0;
    forAll(cellSources_, idxI)
    {
        const label& cellI = cellSources_[idxI];
        if (ksAggregationType_ == "continuous")
        {
            objValTmp += exp(coeffKS_ * scale_ * (tmpField[cellI] - maxValue)) * mesh_.V()[cellI];
        }
        else if (ksAggregationType_ == "discrete")
        {
            objValTmp += exp(coeffKS_ * scale_ * (tmpField[cellI] - maxValue));
        }
        else
        {
            FatalErrorIn(" ") << "KS function ksAggregationType must be continuous or discrete! " << abort(FatalError);
        }

        if (objValTmp > 1e200)
        {
            std::cout << "Warning! procI: " << Pstream::myProcNo() << "\n" << 
                "KS function summation term too large! Reduce coeffKS!  " << std::endl;
            std::cout << "bounded value in this proc to 1e200" << std::endl;
            objValTmp = 1e200;
        }
    }

    // need to reduce the sum of force across all processors
    reduce(objValTmp, sumOp<scalar>());

    // weight by total Volume;
    objValTmp /= alpha_;

    functionValue = log(objValTmp) / coeffKS_ + scale_ * maxValue;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
