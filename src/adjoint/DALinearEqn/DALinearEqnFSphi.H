/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Solve the linear euqation using Petsc

\*---------------------------------------------------------------------------*/

#ifndef DALinearEqnFSphi_H
#define DALinearEqnFSphi_H

#include "DALinearEqn.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DALinearEqn Declaration
\*---------------------------------------------------------------------------*/

class DALinearEqnFSphi
    : public DALinearEqn
{

private:
    /// Disallow default bitwise copy construct
    DALinearEqnFSphi(const DALinearEqnFSphi&);

    /// Disallow default bitwise assignment
    void operator=(const DALinearEqnFSphi&);

public:
    TypeName("FSphi");

    /// Constructors
    DALinearEqnFSphi(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAIndex& daIndex);

    /// Destructor
    virtual ~DALinearEqnFSphi()
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
