/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Solve the linear euqation using Petsc

\*---------------------------------------------------------------------------*/

#ifndef DALinearEqnFSModel_H
#define DALinearEqnFSModel_H

#include "DALinearEqn.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DALinearEqn Declaration
\*---------------------------------------------------------------------------*/

class DALinearEqnFSModel
    : public DALinearEqn
{

private:
    /// Disallow default bitwise copy construct
    DALinearEqnFSModel(const DALinearEqnFSModel&);

    /// Disallow default bitwise assignment
    void operator=(const DALinearEqnFSModel&);

public:
    TypeName("FSModel");

    /// Constructors
    DALinearEqnFSModel(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAIndex& daIndex);

    /// Destructor
    virtual ~DALinearEqnFSModel()
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
