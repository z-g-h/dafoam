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

    // check discipline
    discipline_ = daOption_.getAllOptions().getWord("discipline");

    // check coupling mode and validate
    distanceMode_ = daOption_.getAllOptions().getWord("wallDistanceMethod");
    if (distanceMode_ != "daCustom" && distanceMode_ != "default")
    {
        FatalErrorIn(" ") << "wallDistanceMethod: "
                          << distanceMode_ << " not supported!"
                          << " Options are: default and daCustom."
                          << abort(FatalError);
    }

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

    volScalarField& T =
        const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>("T"));

    // ********* first loop, set the Tn, discipline and distanceMode
    label counterI = 0;
    forAll(patches_, idxI)
    {
        // get the patch id label
        word patchName = patches_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);

        aeroThermalCoupleFvPatchScalarField& aeroThermalCouplePatch =
            refCast<aeroThermalCoupleFvPatchScalarField>(T.boundaryFieldRef()[patchI]);

        // set discipline and distanceMode
        aeroThermalCouplePatch.refDiscipline() = discipline_;
        aeroThermalCouplePatch.refDistanceMode() = distanceMode_;

        forAll(aeroThermalCouplePatch.refTn(), faceI)
        {
            aeroThermalCouplePatch.refTn()[faceI] = input[counterI];
            counterI++;
        }
    }

    // ********* second loop, set the Cn
    forAll(patches_, idxI)
    {
        // get the patch id label
        word patchName = patches_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);

        aeroThermalCoupleFvPatchScalarField& aeroThermalCouplePatch =
            refCast<aeroThermalCoupleFvPatchScalarField>(T.boundaryFieldRef()[patchI]);

        forAll(aeroThermalCouplePatch.refCn(), faceI)
        {
            aeroThermalCouplePatch.refCn()[faceI] = input[counterI];
            counterI++;
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
