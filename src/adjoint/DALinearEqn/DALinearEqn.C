/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DALinearEqn.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DALinearEqn, 0);
defineRunTimeSelectionTable(DALinearEqn, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DALinearEqn::DALinearEqn(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAIndex& daIndex)
    : mesh_(mesh),
      daOption_(daOption),
      daIndex_(daIndex)
{
    KSPCalcSingularVal_ = daOption_.getSubDictOption<label>("adjEqnOption", "KSPCalcSingularVal");
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
autoPtr<DALinearEqn> DALinearEqn::New(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAIndex& daIndex)
{
    // standard setup for runtime selectable classes

    // look up the solver name defined in pyOptions
    word modelType = daOption.getOption<word>("DAPC");

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DAPC" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DALinearEqn::New"
            "("
            "    mesh,"
            "    daOption"
            ")")
            << "Unknown DAPC type "
            << modelType << nl << nl
            << "Valid DASolver types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DALinearEqn>(
        cstrIter()(mesh, daOption, daIndex));
}

label DALinearEqn::solveLinearEqn(
    const KSP ksp,
    const Vec rhsVec,
    Vec solVec)
{
    /*
    Description:
        Solve a linear equation.
    
    Input:
        ksp: the KSP object, obtained from calling Foam::createMLRKSP

        rhsVec: the right-hand-side petsc vector

    Output:
        solVec: the solution vector

        Return 0 if the linear equation solution finished successfully otherwise return 1
    */

    Info << "Solving Linear Equation... " << this->getRunTime() << " s" << endl;

    //Solve adjoint
    // VecZeroEntries(solVec);

    // set up rGMRESHist to save the tolerance history for the GMRES solution
    // these vars are for store the tolerance for GMRES linear solution
    label gmresMaxIters = daOption_.getSubDictOption<label>("adjEqnOption", "gmresMaxIters");
    PetscScalar rGMRESHist[gmresMaxIters + 1];
    label nGMRESIters = gmresMaxIters + 1;
    KSPSetResidualHistory(ksp, rGMRESHist, nGMRESIters, PETSC_TRUE);

    // solve KSP
    KSPSolve(ksp, rhsVec, solVec);

    //Print convergence information
    label its;
    KSPGetIterationNumber(ksp, &its);
    PetscScalar initResNorm = rGMRESHist[0];
    PetscScalar finalResNorm = rGMRESHist[its];
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    PetscPrintf(
        PETSC_COMM_WORLD,
        "Main iteration %d KSP Residual norm %14.12e %.2f s \n",
        its,
        finalResNorm,
        this->getRunTime());

    label KSPCalcEigen = daOption_.getSubDictOption<label>("adjEqnOption", "KSPCalcEigen");
    if (KSPCalcEigen)
    {
        PetscInt n = its; /* upper bound; GMRES gives since-last-restart */
        PetscReal *er, *ei;
        PetscMalloc2(n, &er, n, &ei);
        PetscInt neig;
        KSPComputeEigenvalues(ksp, n, er, ei, &neig); /* preconditioned operator */

        for (PetscInt i = 0; i < neig; i++)
        {
            PetscPrintf(PETSC_COMM_WORLD, "eig[%d] = %g + %g i\n", (int)i, (double)er[i], (double)ei[i]);
        }
        PetscFree2(er, ei);
    }

    if (KSPCalcSingularVal_)
    {
        PetscReal smin, smax, ratio;
        KSPComputeExtremeSingularValues(ksp, &smax, &smin);
        ratio = smax / smin;
        PetscPrintf(PETSC_COMM_WORLD, "Singular value ratio. sMax / sMin = %g / %g = %g \n", (double)smax, (double)smin, (double)ratio);
    }

    Info << "**Completed**! Total iterations: " << its
         << ". PetscConvergedReason: " << reason << ". " << this->getRunTime() << " s" << endl;

    VecAssemblyBegin(solVec);
    VecAssemblyEnd(solVec);

    // now we need to check if the linear equation solution is successful

    scalar absResRatio = finalResNorm / daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresAbsTol");
    scalar relResRatio = finalResNorm / initResNorm / daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresRelTol");
    scalar resDiff = daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresTolDiff");
    if (relResRatio > resDiff && absResRatio > resDiff)
    {
        Info << "Residual tolerance not satisfied, solution failed!" << endl;
        return 1;
    }
    else
    {
        Info << "Residual tolerance satisfied, solution finished!" << endl;
        return 0;
    }

    return 1;
}

PetscErrorCode DALinearEqn::myKSPMonitor(
    KSP ksp,
    PetscInt n,
    PetscReal rnorm,
    void* ctx)
{

    /*
    Descripton:
        Write the solution vector and residual norm to stdout.
        - PetscPrintf() handles output for multiprocessor jobs
        by printing from only one processor in the communicator.
        - The parallel viewer PETSC_VIEWER_STDOUT_WORLD handles
        data from multiple processors so that the output
        is not jumbled.
    */

    DALinearEqn* daLinearEqn = (DALinearEqn*)ctx;

    // residual print frequency
    PetscInt printFrequency = daLinearEqn->getPrintInterval();
    PetscScalar runTime = daLinearEqn->getRunTime();
    if (n % printFrequency == 0)
    {

        PetscPrintf(
            PETSC_COMM_WORLD,
            "Main iteration %D KSP Residual norm %14.12e %.2f s. ",
            n,
            rnorm,
            runTime);

        label KSPCalcSingularVal = daLinearEqn->KSPCalcSingularVal();
        if (KSPCalcSingularVal)
        {
            PetscReal smin, smax, ratio;
            KSPComputeExtremeSingularValues(ksp, &smax, &smin);
            ratio = smax / smin;
            PetscPrintf(PETSC_COMM_WORLD, "sMax/sMin=%g/%g=%g", (double)smax, (double)smin, (double)ratio);
        }

        PetscPrintf(
            PETSC_COMM_WORLD,
            "\n");
    }
    return 0;
}

double DALinearEqn::getRunTime()
{
    /*
    Descripton:
        Return the runtime
    */
    return mesh_.time().elapsedCpuTime();
}

label DALinearEqn::getPrintInterval()
{
    /*
    Descripton:
        Return the printInterval from DAOption
    */
    return daOption_.getOption<label>("printInterval");
}

} // End namespace Foam

// ************************************************************************* //
