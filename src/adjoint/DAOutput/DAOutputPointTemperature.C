/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAOutputPointTemperature.H"
#include "DAMacroFunctions.H"
#include "Pstream.H"
#include "emptyFvPatch.H"
#include "globalIndex.H"
#include "ops.H"
#include "pointConstraints.H"
#include "processorFvPatch.H"
#include "surfaceFields.H"
#include "volFields.H"

#include <cmath>

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
    fieldName_ = outputDict.lookupOrDefault<word>("fieldName", "T");
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

    // Refresh the temperature boundary immediately before evaluating the
    // differentiable point-temperature interpolation.
    const objectRegistry& db = mesh_.thisDb();
    volScalarField& T =
        const_cast<volScalarField&>(
            db.lookupObject<volScalarField>(fieldName_));

    // Mesh coordinates may have been updated after the state vector.  Refresh
    // the temperature boundary in C++ immediately before interpolation so the
    // primal and reverse paths use the same X -> q -> boundary-T state.
    T.correctBoundaryConditions();

    // One C++ operator defines both primal and reverse.  This eliminates a
    // native-primal/custom-adjoint software inconsistency.
    scalarList rawOutput(localRawSize(), 0.0);
    runLocalRaw(rawOutput);
    assemblePointTemperature(
        rawOutput,
        TRef_,
        output);
}

label DAOutputPointTemperature::jacobianOutputSize()
{
    return localRawSize();
}

label DAOutputPointTemperature::jacobianRegisteredOutputSize()
{
    // The first two blocks are active N_local and D_local. The final block
    // contains passive global point IDs and must not be registered.
    return 2 * size_;
}

void DAOutputPointTemperature::runForJacobian(scalarList& output)
{
    runLocalRaw(output);
}

void DAOutputPointTemperature::setJacobianOutputSeeds(scalarList& output, const double* seed)
{
#ifdef CODI_ADR
    List<double> numeratorSeed;
    List<double> denominatorSeed;
    calcReverseSeeds(
        output,
        seed,
        numeratorSeed,
        denominatorSeed);

    for (label pointI = 0; pointI < size_; pointI++)
    {
        output[pointI].setGradient(numeratorSeed[pointI]);
        output[size_ + pointI].setGradient(
            denominatorSeed[pointI]);
    }
#else
    // Only the reverse-AD build calls this method.
    (void)output;
    (void)seed;
#endif
}

void DAOutputPointTemperature::runLocalRaw(scalarList& rawOutput)
{
    if (rawOutput.size() != localRawSize())
    {
        FatalErrorInFunction
            << "Raw output size " << rawOutput.size()
            << " does not match 3*mesh.nPoints() = " << localRawSize()
            << exit(FatalError);
    }

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>(fieldName_);

    const pointField& points = mesh_.points();
    const vectorField& cellCentres = mesh_.cellCentres();
    const vectorField& faceCentres = mesh_.faceCentres();
    const labelListList& pointCells = mesh_.pointCells();
    const fvBoundaryMesh& fvBoundary = mesh_.boundary();
    const polyBoundaryMesh& polyBoundary = mesh_.boundaryMesh();

    // The conditional explicitly communicates the non-negative allocation
    // range to GCC's static range analysis.
    const label nPoints = size_ > 0 ? size_ : 0;

    // Reproduce volPointInterpolation's local physical-patch classification.
    // In the normal debug=0 path this flag is not synchronised across
    // processor boundaries.
    boolList isPatchPoint(nPoints, false);
    forAll(polyBoundary, patchI)
    {
        const polyPatch& patch = polyBoundary[patchI];
        const bool isEmpty = isA<emptyFvPatch>(fvBoundary[patchI]);
        const bool isCoupled = T.boundaryField()[patchI].coupled();

        if (isCoupled && !isA<processorFvPatch>(fvBoundary[patchI]))
        {
            FatalErrorInFunction
                << "pointTemperatureOutput supports processor coupling "
                << "but not separated cyclic/AMI point transforms. "
                << "Unsupported patch: " << patch.name()
                << exit(FatalError);
        }

        if (!isEmpty && !isCoupled)
        {
            forAll(patch, faceI)
            {
                const face& patchFace = patch[faceI];
                forAll(patchFace, facePointI)
                {
                    isPatchPoint[patchFace[facePointI]] = true;
                }
            }
        }
    }

    scalarList numerator(nPoints, 0.0);
    scalarList denominator(nPoints, 0.0);

    // Internal and processor-only points: local cell-centre contributions.
    forAll(pointCells, pointI)
    {
        if (!isPatchPoint[pointI])
        {
            const labelList& cells = pointCells[pointI];
            forAll(cells, pointCellI)
            {
                const label cellI = cells[pointCellI];
                const scalar weight =
                    1.0 / mag(points[pointI] - cellCentres[cellI]);
                numerator[pointI] += weight * T[cellI];
                denominator[pointI] += weight;
            }
        }
    }

    // Physical patch points: local boundary-face contributions. This retains
    // the direct q_conduct -> boundary T -> point T derivative.
    forAll(polyBoundary, patchI)
    {
        const polyPatch& patch = polyBoundary[patchI];
        const bool isEmpty = isA<emptyFvPatch>(fvBoundary[patchI]);
        const bool isCoupled = T.boundaryField()[patchI].coupled();

        if (!isEmpty && !isCoupled)
        {
            forAll(patch, faceI)
            {
                const label globalFaceI = patch.start() + faceI;
                const face& patchFace = patch[faceI];
                const scalar boundaryT = T.boundaryField()[patchI][faceI];

                forAll(patchFace, facePointI)
                {
                    const label pointI = patchFace[facePointI];
                    const scalar weight =
                        1.0 / mag(points[pointI] - faceCentres[globalFaceI]);
                    numerator[pointI] += weight * boundaryT;
                    denominator[pointI] += weight;
                }
            }
        }
    }

    // Give every local point an initially unique global label, then let
    // OpenFOAM's own collocated-point topology replace all processor copies by
    // their minimum label.  Labels are passive, so this synchronization does
    // not cross an MPI boundary with a CoDiPack active scalar.
    globalIndex globalPoints(nPoints);
    labelList pointToGlobal(nPoints);
    forAll(pointToGlobal, pointI)
    {
        pointToGlobal[pointI] = globalPoints.toGlobal(pointI);
    }
    pointConstraints::syncUntransformedData(
        mesh_,
        pointToGlobal,
        minEqOp<label>());
    const label nGlobalPoints = globalPoints.size();

    for (label pointI = 0; pointI < nPoints; pointI++)
    {
        const label globalPointI = pointToGlobal[pointI];
        if (globalPointI < 0 || globalPointI >= nGlobalPoints)
        {
            FatalErrorInFunction
                << "Invalid global point ID " << globalPointI
                << " for local point " << pointI
                << "; global point count is " << nGlobalPoints
                << exit(FatalError);
        }

        rawOutput[pointI] = numerator[pointI];
        rawOutput[nPoints + pointI] = denominator[pointI];
        rawOutput[2 * nPoints + pointI] = scalar(globalPointI);
    }
}

void DAOutputPointTemperature::unpackAndGlobalSum(
    const scalarList& rawOutput,
    labelList& pointToGlobal,
    List<double>& globalNumerator,
    List<double>& globalDenominator)
{
    if (rawOutput.size() % 3 != 0)
    {
        FatalErrorInFunction
            << "Raw point-temperature output size " << rawOutput.size()
            << " is not divisible by three."
            << exit(FatalError);
    }

    const label nPoints = rawOutput.size() / 3;
    pointToGlobal.setSize(nPoints);

    label maxGlobalPointI = -1;
    for (label pointI = 0; pointI < nPoints; pointI++)
    {
        double idValue = 0.0;
        assignValueCheckAD(idValue, rawOutput[2 * nPoints + pointI]);
        const label globalPointI =
            static_cast<label>(std::floor(idValue + 0.5));

        if (
            globalPointI < 0
            || std::fabs(idValue - static_cast<double>(globalPointI)) > 1.0e-10)
        {
            FatalErrorInFunction
                << "Invalid global point ID value " << idValue
                << " at local point " << pointI
                << exit(FatalError);
        }

        pointToGlobal[pointI] = globalPointI;
        maxGlobalPointI = max(maxGlobalPointI, globalPointI);
    }

    reduce(maxGlobalPointI, maxOp<label>());
    const label nGlobalPoints = maxGlobalPointI + 1;

    globalNumerator.setSize(nGlobalPoints, 0.0);
    globalDenominator.setSize(nGlobalPoints, 0.0);

    for (label pointI = 0; pointI < nPoints; pointI++)
    {
        double numeratorValue = 0.0;
        double denominatorValue = 0.0;
        assignValueCheckAD(numeratorValue, rawOutput[pointI]);
        assignValueCheckAD(
            denominatorValue,
            rawOutput[nPoints + pointI]);

        const label globalPointI = pointToGlobal[pointI];
        globalNumerator[globalPointI] += numeratorValue;
        globalDenominator[globalPointI] += denominatorValue;
    }

    if (Pstream::parRun())
    {
        // These are deliberately passive doubles. No CoDiPack scalar crosses
        // MPI; calcReverseSeeds supplies the exact transpose explicitly.
        Pstream::listCombineGather(
            globalNumerator,
            plusEqOp<double>());
        Pstream::listCombineScatter(globalNumerator);

        Pstream::listCombineGather(
            globalDenominator,
            plusEqOp<double>());
        Pstream::listCombineScatter(globalDenominator);
    }
}

void DAOutputPointTemperature::assemblePointTemperature(
    const scalarList& rawOutput,
    const scalar TRef,
    scalarList& pointTemperature)
{
    const label nPoints = rawOutput.size() / 3;
    if (pointTemperature.size() != nPoints)
    {
        FatalErrorInFunction
            << "Point-temperature output size " << pointTemperature.size()
            << " does not match local point count " << nPoints
            << exit(FatalError);
    }

    labelList pointToGlobal;
    List<double> globalNumerator;
    List<double> globalDenominator;
    unpackAndGlobalSum(
        rawOutput,
        pointToGlobal,
        globalNumerator,
        globalDenominator);

    double TRefValue = 0.0;
    assignValueCheckAD(TRefValue, TRef);

    for (label pointI = 0; pointI < nPoints; pointI++)
    {
        const label globalPointI = pointToGlobal[pointI];
        const double denominator = globalDenominator[globalPointI];
        if (
            !std::isfinite(denominator)
            || std::fabs(denominator) <= 1.0e-300)
        {
            FatalErrorInFunction
                << "Invalid interpolation denominator " << denominator
                << " for global point " << globalPointI
                << exit(FatalError);
        }

        pointTemperature[pointI] =
            globalNumerator[globalPointI] / denominator - TRefValue;
    }
}

void DAOutputPointTemperature::calcReverseSeeds(
    const scalarList& rawOutput,
    const double* pointTemperatureSeed,
    List<double>& numeratorSeed,
    List<double>& denominatorSeed)
{
    const label nPoints = rawOutput.size() / 3;

    labelList pointToGlobal;
    List<double> globalNumerator;
    List<double> globalDenominator;
    unpackAndGlobalSum(
        rawOutput,
        pointToGlobal,
        globalNumerator,
        globalDenominator);

    List<double> globalPointSeed(globalNumerator.size(), 0.0);
    for (label pointI = 0; pointI < nPoints; pointI++)
    {
        globalPointSeed[pointToGlobal[pointI]] +=
            pointTemperatureSeed[pointI];
    }

    if (Pstream::parRun())
    {
        Pstream::listCombineGather(
            globalPointSeed,
            plusEqOp<double>());
        Pstream::listCombineScatter(globalPointSeed);
    }

    numeratorSeed.setSize(nPoints, 0.0);
    denominatorSeed.setSize(nPoints, 0.0);

    for (label pointI = 0; pointI < nPoints; pointI++)
    {
        const label globalPointI = pointToGlobal[pointI];
        const double denominator = globalDenominator[globalPointI];
        if (
            !std::isfinite(denominator)
            || std::fabs(denominator) <= 1.0e-300)
        {
            FatalErrorInFunction
                << "Invalid interpolation denominator " << denominator
                << " for global point " << globalPointI
                << exit(FatalError);
        }

        const double seed = globalPointSeed[globalPointI];
        numeratorSeed[pointI] = seed / denominator;
        denominatorSeed[pointI] =
            -seed * globalNumerator[globalPointI]
            / (denominator * denominator);
    }
}

} // End namespace Foam
