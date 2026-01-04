/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Solve the linear euqation using Petsc

\*---------------------------------------------------------------------------*/

#ifndef DALinearEqnGAMG_H
#define DALinearEqnGAMG_H

#include "DALinearEqn.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DALinearEqn Declaration
\*---------------------------------------------------------------------------*/

class DALinearEqnGAMG
    : public DALinearEqn
{

private:
    /// Disallow default bitwise copy construct
    DALinearEqnGAMG(const DALinearEqnGAMG&);

    /// Disallow default bitwise assignment
    void operator=(const DALinearEqnGAMG&);

public:
    TypeName("GAMG");

    /// Constructors
    DALinearEqnGAMG(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAIndex& daIndex);

    /// Destructor
    virtual ~DALinearEqnGAMG()
    {
    }

    // Members

    /// create a multi-level, Richardson KSP object
    virtual void createMLRKSP(
        const Mat jacMat,
        const Mat jacPCMat,
        KSP ksp);

    void setPCASM(KSP ksp);
    
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
