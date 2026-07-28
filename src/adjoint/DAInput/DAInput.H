/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Assign the input array to the OF variables

\*---------------------------------------------------------------------------*/

#ifndef DAInput_H
#define DAInput_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAStateInfo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAInput Declaration
\*---------------------------------------------------------------------------*/

class DAInput
{

private:
    /// Disallow default bitwise copy construct
    DAInput(const DAInput&);

    /// Disallow default bitwise assignment
    void operator=(const DAInput&);

protected:
    /// name of the input
    const word inputName_;

    /// type of the input
    const word inputType_;

    /// fvMesh
    fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAIndex object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

public:
    /// Runtime type information
    TypeName("DAInput");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAInput,
        dictionary,
        (
            const word inputName,
            const word inputType,
            fvMesh& mesh,
            const DAOption& daOption,
            const DAModel& daModel,
            const DAIndex& daIndex),
        (
            inputName,
            inputType,
            mesh,
            daOption,
            daModel,
            daIndex));

    // Constructors

    //- Construct from components
    DAInput(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAInput> New(
        const word inputName,
        const word inputType,
        fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAInput()
    {
    }

    virtual void run(const scalarList& input) = 0;

    virtual label size() = 0;

    virtual label distributed() = 0;

    // ---------------------------------------------------------------------
    // Generic reverse-AD input interface.
    //
    // Most inputs use the physical input array directly, so the defaults
    // below are equivalent to size(), run(), the historical DASolver input
    // registration loop, and the historical gradient extraction loop.
    // Inputs containing passive mappings or special post-processing may
    // override these methods without adding input-specific logic to DASolver.
    // ---------------------------------------------------------------------

    /// Number of scalar entries allocated for reverse-AD tape input.
    virtual label jacobianInputSize();

    /// Number of allocated entries registered as CoDiPack tape inputs.
    virtual label jacobianRegisteredInputSize();

    /// Convert the physical input array to the representation recorded on tape.
    virtual void setJacobianInputValues(
        scalarList& jacobianInput,
        const double* physicalInput);

    /// Apply the input representation while recording or cleaning the tape.
    virtual void runForJacobian(
        const scalarList& jacobianInput);

    /// Convert tape-input gradients to the physical Jacobian-vector product.
    virtual void getJacobianInputProduct(
        const scalarList& jacobianInput,
        double* product);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
