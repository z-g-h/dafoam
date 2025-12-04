/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Solve the linear euqation using Petsc

\*---------------------------------------------------------------------------*/

#ifndef DALinearEqnSOR_H
#define DALinearEqnSOR_H

#include "DALinearEqn.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DALinearEqn Declaration
\*---------------------------------------------------------------------------*/

class DALinearEqnSOR
    : public DALinearEqn
{

private:
    /// Disallow default bitwise copy construct
    DALinearEqnSOR(const DALinearEqnSOR&);

    /// Disallow default bitwise assignment
    void operator=(const DALinearEqnSOR&);

public:
    TypeName("SOR");

    /// Constructors
    DALinearEqnSOR(
        const fvMesh& mesh,
        const DAOption& daOption);

    /// Destructor
    virtual ~DALinearEqnSOR()
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
