/*---------------------------------------------------------------------------*\
    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for CHT coupling inputs.

        The input layout is

            [Tn for all coupling faces, Cn for all coupling faces],

        where Cn = kappa_neighbour/d_neighbour. This class only transfers Tn
        and Cn to aeroThermalCoupleFvPatchScalarField. The boundary condition
        computes refValue, refGrad, and

            valueFraction = Cn / (Cn + Clocal)

        whenever T.correctBoundaryConditions() is evaluated. Keeping this
        derived calculation inside the boundary condition allows the AD tape
        to record the local state and mesh dependencies of Clocal.
\*---------------------------------------------------------------------------*/

#ifndef DAInputThermalCoupling_H
#define DAInputThermalCoupling_H

#include "DAInput.H"
#include "addToRunTimeSelectionTable.H"
#include "aeroThermalCoupleFvPatchScalarField.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAInputThermalCoupling Declaration
\*---------------------------------------------------------------------------*/

class DAInputThermalCoupling
    : public DAInput
{

protected:
    /// list of patch names for the thermal var
    wordList patches_;

    /// the total face number for all the patches_ owned by this processor
    label size_;

    /// for distanceMode and discipline is get from daoptions directly.

public:
    TypeName("thermalCouplingInput");
    // Constructors

    //- Construct from components
    DAInputThermalCoupling(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInputThermalCoupling()
    {
    }

    virtual void run(const scalarList& input);

    virtual label size()
    {
        return size_;
    }

    virtual label distributed()
    {
        return 1;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
