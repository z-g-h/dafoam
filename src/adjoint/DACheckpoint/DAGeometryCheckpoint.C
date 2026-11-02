/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAGeometryCheckpoint.H"
#include "coupledPolyPatch.H"
#include "cyclicAMIPolyPatch.H"
#include "processorPolyPatch.H"

namespace Foam
{

#ifdef CODI_ADR

namespace
{

template<class Operation>
void applyScalarFieldValues(scalarField& field, Operation& operation)
{
    forAll(field, idxI)
    {
        operation(field[idxI]);
    }
}

template<class Operation>
void applyScalarListValues(scalarList& values, Operation& operation)
{
    forAll(values, idxI)
    {
        operation(values[idxI]);
    }
}

template<class Operation>
void applyScalarListListValues(scalarListList& values, Operation& operation)
{
    forAll(values, faceI)
    {
        applyScalarListValues(values[faceI], operation);
    }
}

template<class Operation>
void applyVectorFieldValues(vectorField& field, Operation& operation)
{
    forAll(field, idxI)
    {
        for (label cmpt = 0; cmpt < 3; cmpt++)
        {
            operation(field[idxI][cmpt]);
        }
    }
}

template<class Operation>
void applyTensorFieldValues(tensorField& field, Operation& operation)
{
    forAll(field, idxI)
    {
        for (direction cmpt = 0; cmpt < tensor::nComponents; cmpt++)
        {
            operation(field[idxI][cmpt]);
        }
    }
}

template<class Operation>
void applyVolVectorFieldValues(volVectorField& field, Operation& operation)
{
    forAll(field, cellI)
    {
        for (label cmpt = 0; cmpt < 3; cmpt++)
        {
            operation(field[cellI][cmpt]);
        }
    }

    forAll(field.boundaryFieldRef(), patchI)
    {
        forAll(field.boundaryFieldRef()[patchI], faceI)
        {
            for (label cmpt = 0; cmpt < 3; cmpt++)
            {
                operation(field.boundaryFieldRef()[patchI][faceI][cmpt]);
            }
        }
    }
}

template<class Operation>
void applySurfaceVectorFieldValues(surfaceVectorField& field, Operation& operation)
{
    forAll(field, faceI)
    {
        for (label cmpt = 0; cmpt < 3; cmpt++)
        {
            operation(field[faceI][cmpt]);
        }
    }

    forAll(field.boundaryFieldRef(), patchI)
    {
        forAll(field.boundaryFieldRef()[patchI], faceI)
        {
            for (label cmpt = 0; cmpt < 3; cmpt++)
            {
                operation(field.boundaryFieldRef()[patchI][faceI][cmpt]);
            }
        }
    }
}

template<class Operation>
void applySurfaceScalarFieldValues(surfaceScalarField& field, Operation& operation)
{
    forAll(field, faceI)
    {
        operation(field[faceI]);
    }

    forAll(field.boundaryFieldRef(), patchI)
    {
        forAll(field.boundaryFieldRef()[patchI], faceI)
        {
            operation(field.boundaryFieldRef()[patchI][faceI]);
        }
    }
}

/*
 * Visit every persistent geometry value that a finite-volume operator or
 * coupled boundary condition can read. The same order is used for counting,
 * tape registration, gradient transfer, and deactivation.
 */
template<class Operation>
void applyGeometryValues(fvMesh& mesh, Operation& operation)
{
    pointField& points = const_cast<pointField&>(mesh.points());
    vectorField& cellCentres = const_cast<vectorField&>(mesh.cellCentres());
    vectorField& faceCentres = const_cast<vectorField&>(mesh.faceCentres());
    scalarField& cellVolumes = const_cast<scalarField&>(mesh.cellVolumes());
    vectorField& faceAreas = const_cast<vectorField&>(mesh.faceAreas());

    applyVectorFieldValues(points, operation);
    applyVectorFieldValues(cellCentres, operation);
    applyVectorFieldValues(faceCentres, operation);
    applyScalarFieldValues(cellVolumes, operation);
    applyVectorFieldValues(faceAreas, operation);

    const polyBoundaryMesh& boundaryMesh = mesh.boundaryMesh();
    forAll(boundaryMesh, patchI)
    {
        const polyPatch& patch = boundaryMesh[patchI];
        if (isA<processorPolyPatch>(patch))
        {
            const processorPolyPatch& processorPatch =
                refCast<const processorPolyPatch>(patch);

            vectorField& neighbFaceCentres = const_cast<vectorField&>
            (
                processorPatch.neighbFaceCentres()
            );
            vectorField& neighbFaceAreas = const_cast<vectorField&>
            (
                processorPatch.neighbFaceAreas()
            );
            vectorField& neighbFaceCellCentres = const_cast<vectorField&>
            (
                processorPatch.neighbFaceCellCentres()
            );

            applyVectorFieldValues(neighbFaceCentres, operation);
            applyVectorFieldValues(neighbFaceAreas, operation);
            applyVectorFieldValues(neighbFaceCellCentres, operation);
        }

        if (isA<coupledPolyPatch>(patch))
        {
            const coupledPolyPatch& coupledPatch =
                refCast<const coupledPolyPatch>(patch);

            tensorField& forwardT = const_cast<tensorField&>
            (
                coupledPatch.forwardT()
            );
            tensorField& reverseT = const_cast<tensorField&>
            (
                coupledPatch.reverseT()
            );
            vectorField& separation = const_cast<vectorField&>
            (
                coupledPatch.separation()
            );

            applyTensorFieldValues(forwardT, operation);
            applyTensorFieldValues(reverseT, operation);
            applyVectorFieldValues(separation, operation);
        }

        if (isA<cyclicAMIPolyPatch>(patch))
        {
            const cyclicAMIPolyPatch& cyclicAMIPatch =
                refCast<const cyclicAMIPolyPatch>(patch);

            if (cyclicAMIPatch.owner())
            {
                AMIPatchToPatchInterpolation& AMI =
                    const_cast<AMIPatchToPatchInterpolation&>
                    (
                        cyclicAMIPatch.AMI()
                    );

                applyScalarListValues(AMI.srcMagSf(), operation);
                applyScalarListValues(AMI.tgtMagSf(), operation);
                applyScalarListListValues(AMI.srcWeights(), operation);
                applyScalarListListValues(AMI.tgtWeights(), operation);
                applyScalarFieldValues(AMI.srcWeightsSum(), operation);
                applyScalarFieldValues(AMI.tgtWeightsSum(), operation);
            }
        }
    }

    DimensionedField<scalar, volMesh>& V =
        const_cast<DimensionedField<scalar, volMesh>&>(mesh.V());
    forAll(V, cellI)
    {
        operation(V[cellI]);
    }

    volVectorField& C = const_cast<volVectorField&>(mesh.C());
    surfaceVectorField& Cf = const_cast<surfaceVectorField&>(mesh.Cf());
    surfaceVectorField& Sf = const_cast<surfaceVectorField&>(mesh.Sf());
    surfaceScalarField& magSf = const_cast<surfaceScalarField&>(mesh.magSf());
    surfaceScalarField& weights = const_cast<surfaceScalarField&>(mesh.weights());
    surfaceScalarField& deltaCoeffs =
        const_cast<surfaceScalarField&>(mesh.deltaCoeffs());
    surfaceScalarField& nonOrthDeltaCoeffs =
        const_cast<surfaceScalarField&>(mesh.nonOrthDeltaCoeffs());
    surfaceVectorField& nonOrthCorrectionVectors =
        const_cast<surfaceVectorField&>(mesh.nonOrthCorrectionVectors());

    applyVolVectorFieldValues(C, operation);
    applySurfaceVectorFieldValues(Cf, operation);
    applySurfaceVectorFieldValues(Sf, operation);
    applySurfaceScalarFieldValues(magSf, operation);
    applySurfaceScalarFieldValues(weights, operation);
    applySurfaceScalarFieldValues(deltaCoeffs, operation);
    applySurfaceScalarFieldValues(nonOrthDeltaCoeffs, operation);
    applySurfaceVectorFieldValues(nonOrthCorrectionVectors, operation);
}

struct GeometryValueCounter
{
    label size;

    GeometryValueCounter()
        : size(0)
    {
    }

    void operator()(scalar& value)
    {
        (void)value;
        size++;
    }
};

struct GeometryInputRegistrar
{
    codi::RealReverse::Tape& tape;

    explicit GeometryInputRegistrar(codi::RealReverse::Tape& tapeIn)
        : tape(tapeIn)
    {
    }

    void operator()(scalar& value)
    {
        tape.registerInput(value);
    }
};

struct GeometryOutputRegistrar
{
    codi::RealReverse::Tape& tape;

    explicit GeometryOutputRegistrar(codi::RealReverse::Tape& tapeIn)
        : tape(tapeIn)
    {
    }

    void operator()(scalar& value)
    {
        tape.registerOutput(value);
    }
};

struct GeometryGradientReader
{
    List<double>& gradients;
    label index;

    explicit GeometryGradientReader(List<double>& gradientsIn)
        : gradients(gradientsIn),
          index(0)
    {
    }

    void operator()(scalar& value)
    {
        gradients[index++] = value.getGradient();
    }
};

struct GeometryGradientSeeder
{
    const List<double>& gradients;
    label index;

    explicit GeometryGradientSeeder(const List<double>& gradientsIn)
        : gradients(gradientsIn),
          index(0)
    {
    }

    void operator()(scalar& value)
    {
        value.setGradient(gradients[index++]);
    }
};

struct GeometryValueDeactivator
{
    codi::RealReverse::Tape& tape;

    explicit GeometryValueDeactivator(codi::RealReverse::Tape& tapeIn)
        : tape(tapeIn)
    {
    }

    void operator()(scalar& value)
    {
        tape.deactivateValue(value);
    }
};

void checkAMIAddressingSide(
    const labelListList& currentAddress,
    const labelListList& referenceAddress,
    const word& patchName,
    const word& sideName)
{
    if (currentAddress.size() != referenceAddress.size())
    {
        FatalErrorInFunction
            << "AMI patch " << patchName << " changed its " << sideName
            << " addressing size from " << referenceAddress.size()
            << " to " << currentAddress.size()
            << " during geometry-checkpoint replay."
            << abort(FatalError);
    }

    forAll(currentAddress, faceI)
    {
        if (currentAddress[faceI].size() != referenceAddress[faceI].size())
        {
            FatalErrorInFunction
                << "AMI patch " << patchName << " changed the number of "
                << sideName << " overlap entries for face " << faceI
                << " from " << referenceAddress[faceI].size()
                << " to " << currentAddress[faceI].size()
                << " during geometry-checkpoint replay."
                << abort(FatalError);
        }

        forAll(currentAddress[faceI], addressI)
        {
            if
            (
                currentAddress[faceI][addressI]
             != referenceAddress[faceI][addressI]
            )
            {
                FatalErrorInFunction
                    << "AMI patch " << patchName << " changed its "
                    << sideName << " addressing at face " << faceI
                    << ", overlap entry " << addressI << " from "
                    << referenceAddress[faceI][addressI] << " to "
                    << currentAddress[faceI][addressI]
                    << " during geometry-checkpoint replay."
                    << abort(FatalError);
            }
        }
    }
}

} // End anonymous namespace

DAGeometryCheckpoint::DAGeometryCheckpoint(fvMesh& mesh)
    : mesh_(mesh),
      referenceSize_(-1)
{
}

void DAGeometryCheckpoint::captureAMIAddressing()
{
    const polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
    label nOwnerAMIPatches = 0;
    forAll(boundaryMesh, patchI)
    {
        const polyPatch& patch = boundaryMesh[patchI];
        if
        (
            isA<cyclicAMIPolyPatch>(patch)
         && refCast<const cyclicAMIPolyPatch>(patch).owner()
        )
        {
            nOwnerAMIPatches++;
        }
    }

    amiAddressingSnapshots_.setSize(nOwnerAMIPatches);
    label snapshotI = 0;
    forAll(boundaryMesh, patchI)
    {
        const polyPatch& patch = boundaryMesh[patchI];
        if (!isA<cyclicAMIPolyPatch>(patch))
        {
            continue;
        }

        const cyclicAMIPolyPatch& cyclicAMIPatch =
            refCast<const cyclicAMIPolyPatch>(patch);
        if (!cyclicAMIPatch.owner())
        {
            continue;
        }

        const AMIPatchToPatchInterpolation& AMI = cyclicAMIPatch.AMI();
        if (Pstream::parRun() && AMI.singlePatchProc() < 0)
        {
            FatalErrorInFunction
                << "Geometry checkpoint currently requires both sides of AMI "
                << "patch " << patch.name()
                << " to reside on one processor, but AMI::singlePatchProc() "
                << "returned " << AMI.singlePatchProc() << "."
                << abort(FatalError);
        }

        AMIAddressingSnapshot& snapshot =
            amiAddressingSnapshots_[snapshotI++];
        snapshot.patchName = patch.name();
        snapshot.singlePatchProc = AMI.singlePatchProc();
        snapshot.srcAddress = AMI.srcAddress();
        snapshot.tgtAddress = AMI.tgtAddress();
    }
}

void DAGeometryCheckpoint::verifyAMIAddressing() const
{
    const polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
    forAll(amiAddressingSnapshots_, snapshotI)
    {
        const AMIAddressingSnapshot& snapshot =
            amiAddressingSnapshots_[snapshotI];
        const label patchI = boundaryMesh.findPatchID(snapshot.patchName);
        if
        (
            patchI < 0
         || !isA<cyclicAMIPolyPatch>(boundaryMesh[patchI])
        )
        {
            FatalErrorInFunction
                << "AMI owner patch " << snapshot.patchName
                << " disappeared or changed type during geometry-checkpoint "
                << "replay."
                << abort(FatalError);
        }

        const cyclicAMIPolyPatch& cyclicAMIPatch =
            refCast<const cyclicAMIPolyPatch>(boundaryMesh[patchI]);
        if (!cyclicAMIPatch.owner())
        {
            FatalErrorInFunction
                << "AMI patch " << snapshot.patchName
                << " is no longer the owner during geometry-checkpoint replay."
                << abort(FatalError);
        }

        const AMIPatchToPatchInterpolation& AMI = cyclicAMIPatch.AMI();
        if (AMI.singlePatchProc() != snapshot.singlePatchProc)
        {
            FatalErrorInFunction
                << "AMI patch " << snapshot.patchName
                << " changed singlePatchProc from " << snapshot.singlePatchProc
                << " to " << AMI.singlePatchProc()
                << " during geometry-checkpoint replay."
                << abort(FatalError);
        }

        checkAMIAddressingSide
        (
            AMI.srcAddress(),
            snapshot.srcAddress,
            snapshot.patchName,
            "source"
        );
        checkAMIAddressingSide
        (
            AMI.tgtAddress(),
            snapshot.tgtAddress,
            snapshot.patchName,
            "target"
        );
    }
}

void DAGeometryCheckpoint::captureReference()
{
    GeometryValueCounter counter;
    applyGeometryValues(mesh_, counter);
    referenceSize_ = counter.size;
    captureAMIAddressing();
}

void DAGeometryCheckpoint::registerInputs(codi::RealReverse::Tape& tape)
{
    GeometryInputRegistrar registrar(tape);
    applyGeometryValues(mesh_, registrar);
}

void DAGeometryCheckpoint::registerOutputs(codi::RealReverse::Tape& tape)
{
    GeometryOutputRegistrar registrar(tape);
    applyGeometryValues(mesh_, registrar);
}

void DAGeometryCheckpoint::readGradients(List<double>& gradients)
{
    if (gradients.size() != referenceSize_)
    {
        FatalErrorInFunction
            << "Geometry-gradient storage has size " << gradients.size()
            << " but the checkpoint interface has size " << referenceSize_
            << "." << abort(FatalError);
    }

    GeometryGradientReader reader(gradients);
    applyGeometryValues(mesh_, reader);
}

void DAGeometryCheckpoint::seedGradients(const List<double>& gradients)
{
    if (gradients.size() != referenceSize_)
    {
        FatalErrorInFunction
            << "Geometry-gradient seed has size " << gradients.size()
            << " but the checkpoint interface has size " << referenceSize_
            << "." << abort(FatalError);
    }

    GeometryGradientSeeder seeder(gradients);
    applyGeometryValues(mesh_, seeder);
}

void DAGeometryCheckpoint::deactivateValues(codi::RealReverse::Tape& tape)
{
    GeometryValueDeactivator deactivator(tape);
    applyGeometryValues(mesh_, deactivator);
}

void DAGeometryCheckpoint::verifyReplay() const
{
    GeometryValueCounter counter;
    applyGeometryValues(mesh_, counter);
    if (counter.size != referenceSize_)
    {
        FatalErrorInFunction
            << "Geometry checkpoint interface changed from " << referenceSize_
            << " to " << counter.size << " values during replay."
            << abort(FatalError);
    }

    verifyAMIAddressing();
}

#endif

} // End namespace Foam

