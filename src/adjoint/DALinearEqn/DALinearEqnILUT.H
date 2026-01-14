/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Solve the linear euqation using Petsc

\*---------------------------------------------------------------------------*/

#ifndef DALinearEqnILUT_H
#define DALinearEqnILUT_H

#include "DALinearEqn.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DALinearEqn Declaration
\*---------------------------------------------------------------------------*/

class DALinearEqnILUT
    : public DALinearEqn
{

private:
    /// Disallow default bitwise copy construct
    DALinearEqnILUT(const DALinearEqnILUT&);

    /// Disallow default bitwise assignment
    void operator=(const DALinearEqnILUT&);

public:
    TypeName("ILUT");

    /// Constructors
    DALinearEqnILUT(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAIndex& daIndex);

    /// Destructor
    virtual ~DALinearEqnILUT()
    {
    }

    // Members

    /// create a multi-level, Richardson KSP object
    virtual void createMLRKSP(
        const Mat jacMat,
        const Mat jacPCMat,
        KSP ksp);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
