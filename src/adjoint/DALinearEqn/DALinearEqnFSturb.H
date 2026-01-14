/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Solve the linear euqation using Petsc

\*---------------------------------------------------------------------------*/

#ifndef DALinearEqnturb_H
#define DALinearEqnturb_H

#include "DALinearEqn.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DALinearEqn Declaration
\*---------------------------------------------------------------------------*/

class DALinearEqnFSturb
    : public DALinearEqn
{

private:
    /// Disallow default bitwise copy construct
    DALinearEqnFSturb(const DALinearEqnFSturb&);

    /// Disallow default bitwise assignment
    void operator=(const DALinearEqnFSturb&);

public:
    TypeName("FSturb");

    /// Constructors
    DALinearEqnFSturb(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAIndex& daIndex);

    /// Destructor
    virtual ~DALinearEqnFSturb()
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
