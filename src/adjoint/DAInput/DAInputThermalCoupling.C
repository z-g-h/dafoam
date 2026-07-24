/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputThermalCoupling.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputThermalCoupling, 0);
addToRunTimeSelectionTable(DAInput, DAInputThermalCoupling, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputThermalCoupling::DAInputThermalCoupling(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAInput(
        inputName,
        inputType,
        mesh,
        daOption,
        daModel,
        daIndex)
{

    daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).readEntry("patches", patches_);
    // NOTE: always sort the patch because the order of the patch element matters in CHT coupling
    sort(patches_);

    size_ = 0;
    forAll(patches_, idxI)
    {
        word patchName = patches_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);
        forAll(mesh_.boundaryMesh()[patchI], faceI)
        {
            size_++;
        }
    }
    // we have two sets of variable to transfer
    size_ *= 2;
}

void DAInputThermalCoupling::run(const scalarList& input)
{
    /*
    Description:
        Assign the input to OF fields
    */

    if (input.size() != size_)
    {
        FatalErrorInFunction
            << "Input '" << inputName_ << "' has size " << input.size()
            << ", but thermalCouplingInput expects " << size_
            << " entries: [Tn for all coupling faces, "
            << "Cn for all coupling faces]."
            << abort(FatalError);
    }

    volScalarField& T =
        const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>("T"));

   // Clear any "already updated" state left by matrix assembly.
    // fvPatchField::evaluate() resets that private flag to false. Therefore
    // the final T.correctBoundaryConditions() is guaranteed to call
    // aeroThermalCouple::updateCoeffs() with the newly assigned Tn and Cn.
    forAll(patches_, idxI)
    {
        const label patchI =
            mesh_.boundaryMesh().findPatchID(patches_[idxI]);

        fvPatchScalarField& patchField = T.boundaryFieldRef()[patchI];

        if (!isA<aeroThermalCoupleFvPatchScalarField>(patchField))
        {
            FatalErrorInFunction
                << "Patch '" << patches_[idxI] << "' in field T has type '"
                << patchField.type() << "', but thermalCouplingInput requires "
                << "type 'aeroThermalCouple'."
                << abort(FatalError);
        }

        aeroThermalCoupleFvPatchScalarField& couplingPatch =
            refCast<aeroThermalCoupleFvPatchScalarField>(patchField);

        if (couplingPatch.updated())
        {
            couplingPatch.evaluate();
        }
    }

    label counterI = 0;

    // First half: neighbour temperature Tn.
    forAll(patches_, idxI)
    {
        const label patchI =
            mesh_.boundaryMesh().findPatchID(patches_[idxI]);

        aeroThermalCoupleFvPatchScalarField& couplingPatch =
            refCast<aeroThermalCoupleFvPatchScalarField>(
                T.boundaryFieldRef()[patchI]);

        forAll(couplingPatch.refTn(), faceI)
        {
            couplingPatch.refTn()[faceI] = input[counterI++];
        }
    }

    // Second half: neighbour conductance Cn = kappa_neighbour/d_neighbour.
    forAll(patches_, idxI)
    {
        const label patchI =
            mesh_.boundaryMesh().findPatchID(patches_[idxI]);

        aeroThermalCoupleFvPatchScalarField& couplingPatch =
            refCast<aeroThermalCoupleFvPatchScalarField>(
                T.boundaryFieldRef()[patchI]);

        forAll(couplingPatch.refCn(), faceI)
        {
            couplingPatch.refCn()[faceI] = input[counterI++];
        }
    }

    // updateCoeffs() owns the complete differentiable relation
    //
    //   f = Cn / (Cn + Clocal),
    //
    // including Clocal(T, X). This is what preserves the missing
    // T -> kappa -> f and X -> deltaCoeffs -> f derivative paths.
    T.correctBoundaryConditions();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
