/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DALinearEqnFS.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DALinearEqnFS, 0);
addToRunTimeSelectionTable(DALinearEqn, DALinearEqnFS, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DALinearEqnFS::DALinearEqnFS(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAIndex& daIndex)
    : DALinearEqn(mesh, daOption, daIndex)
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DALinearEqnFS::createMLRKSP(
    const Mat jacMat,
    const Mat jacPCMat,
    KSP ksp)
{
    /*
    Description:
        This is the main function we need to call to initialize the KSP and set
        up parameters for solving the linear equations
    
    Input:
        gmresRestart: how many Krylov spaces to keep before resetting them.
        Usually, this is set to the gmresMaxIters

        gmresMaxIters: how many GMRES iteration to run at most

        gmresRelTol: the relative tolerance for GMRES

        gmresAbsTol: the absolute tolerance for GMRES

        globalPCIters: globa iteration for PC, usually set it to 0

        asmOverlap: ASM overlap for solving the linearEqn in parallel. 
        Usually set it to 1. Setting a higher number increases the convergence but
        significantly increase the memory usage

        localPCIters: local iteraction for PC. usually set it to 1
        
        jacMatReOrdering: re-order the lhs matrix to reduce memory usage.
        Usually we use nd, rcm, or natural (not re-ordered)
    
        pcFillLevel: how many leve fill-in to use for PC. This is a critical
        parameters for convergence rate. Usually set it to 1. Setting it to a higher
        number increase the convergence, however, the memory usage generally grows 
        exponetially. We rarely set it more than 2.

        printInfo: whether to print summary information before solving 

        jacMat: the right-hand-side petsc matrix 

        jacPCMat: the preconditioner matrix from which we constructor our preconditioners
    
    Output:
        genksp: the set KSP object 
    */

    const label gmresRestart =
        daOption_.getSubDictOption<label>("adjEqnOption", "gmresRestart");
    const label asmOverlap =
        daOption_.getSubDictOption<label>("adjEqnOption", "asmOverlap");
    const word jacMatReOrdering =
        daOption_.getSubDictOption<word>("adjEqnOption", "jacMatReOrdering");
    const label pcFillLevel =
        daOption_.getSubDictOption<label>("adjEqnOption", "pcFillLevel");
    const label gmresMaxIters =
        daOption_.getSubDictOption<label>("adjEqnOption", "gmresMaxIters");
    const scalar gmresRelTol =
        daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresRelTol");
    const scalar gmresAbsTol =
        daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresAbsTol");
    const label useNonZeroInitGuess =
        daOption_.getSubDictOption<label>("adjEqnOption", "useNonZeroInitGuess");
    const label useMGSO =
        daOption_.getSubDictOption<label>("adjEqnOption", "useMGSO");
    const label printInfo =
        daOption_.getSubDictOption<label>("adjEqnOption", "printInfo");

    PC MLRGlobalPC;
    // ASM Preconditioner variables
    // PetscInt MLRoverlap; // width of subdomain overlap

    // Create linear solver context
    //KSPCreate(PETSC_COMM_WORLD, &ksp);

    // Set operators. Here the matrix that defines the linear
    // system also serves as the preconditioning matrix.
    KSPSetOperators(ksp, jacMat, jacPCMat);

    // This code sets up the supplied kspObject in the following
    // specific fashion.
    //
    // The hierarchy of the setup is:
    //  kspObject --> Supplied KSP object
    //  |
    //  --> master_PC --> Preconditioner type set to KSP
    //      |
    //      --> master_PC_KSP --> KSP type set to Richardson with 'globalPreConIts'
    //          |
    //           --> globalPC --> PC type set to 'globalPCType'
    //                  |         Use fieldSplit to split flow variables and model variable.
    //                  |
    //                   --> subPC -->  Usually Additive Schwartz and overlap is set
    //                         |         with 'ASMOverlap'. Use 0 to get BlockJacobi.
    //                         |
    //                     subsubPC-->   Usually ILU. 'localFillLevel' is
    //                                  set and 'localMatrixOrder' is used.
    //

    // First, KSPSetFromOptions MUST be called
    if (daOption_.getOption<label>("debug"))
    {
        PetscOptionsSetValue(NULL, "-ksp_view_eigenvalues", NULL);
    }
    KSPSetFromOptions(ksp);

    // Set GMRES
    // Set the type of solver to GMRES
    KSPType kspObjectType = KSPGMRES;

    KSPSetType(ksp, kspObjectType);
    // Set the gmres restart
    PetscInt restartGMRES = gmresRestart;

    // whether to use non-zero initial guess
    if (useNonZeroInitGuess)
    {
        KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    }
    else
    {
        KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    }

    KSPGMRESSetRestart(ksp, restartGMRES);
    // Set the GMRES refinement type
    KSPGMRESSetCGSRefinementType(ksp, KSP_GMRES_CGS_REFINE_IFNEEDED);

    // set orthogonalization for the GMRES, useMGSO=1: modified Gram Schmidt
    // useMGSO=0: classical Gram Schmidt
    if (useMGSO)
    {
        KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization);
    }

    // Set the preconditioner side
    KSPSetPCSide(ksp, PC_RIGHT);

    KSPGetPC(ksp, &MLRGlobalPC);

    // Set the type of 'MLRGlobalPC'. This will use fieldSplit to split flow variables and model variables
    PCSetType(MLRGlobalPC, PCFIELDSPLIT);

    // get model state name and size
    label modelStateSize = 0;
    List<word> modelState;
    // get surface state name and size
    label surfaceStateSize = 0;
    List<word> surfaceState;
    forAll(daIndex_.adjStateNames, idx)
    {
        word stateName = daIndex_.adjStateNames[idx];
        if (daIndex_.adjStateType[stateName] == "modelState")
        {
            modelStateSize++;
            modelState.append(stateName);
        }
        else if (daIndex_.adjStateType[stateName] == "surfaceScalarState")
        {
            surfaceStateSize++;
            surfaceState.append(stateName);
        }
    }

    // exact model and surface state IS
    IS is_flowState, is_modelState, is_surfaceState;
    const label nLocalCells = daIndex_.nLocalCells;
    const label nLocalFaces = daIndex_.nLocalFaces;
    List<PetscInt> modelStateIndices(modelStateSize * nLocalCells);
    List<PetscInt> surfaceStateIndices(surfaceStateSize * nLocalFaces);

    // get model state index in dRdW
    forAll(mesh_.cells(), cellI)
    {
        for (PetscInt idx = 0; idx < modelStateSize; idx++)
        {
            PetscInt globalIdx = daIndex_.getGlobalAdjointStateIndex(modelState[idx], cellI);
            modelStateIndices[modelStateSize * cellI + idx] = globalIdx;
        }
    }

    // get surface state index in dRdW
    forAll(mesh_.faces(), faceI)
    {
        for (PetscInt idx = 0; idx < surfaceStateSize; idx++)
        {
            PetscInt globalIdx = daIndex_.getGlobalAdjointStateIndex(surfaceState[idx], faceI);
            surfaceStateIndices[surfaceStateSize * faceI + idx] = globalIdx;
        }
    }

    // create IS for model state
    ISCreateGeneral(
        PETSC_COMM_WORLD,
        modelStateIndices.size(),
        modelStateIndices.begin(),
        PETSC_COPY_VALUES,
        &is_modelState);
    ISSort(is_modelState);

    // create IS for surface state
    ISCreateGeneral(
        PETSC_COMM_WORLD,
        surfaceStateIndices.size(),
        surfaceStateIndices.begin(),
        PETSC_COPY_VALUES,
        &is_surfaceState);
    ISSort(is_surfaceState);

    // use a complement to get flow state IS
    IS is_unionArray[2], is_union;
    is_unionArray[0] = is_modelState;
    is_unionArray[1] = is_surfaceState;
    ISConcatenate(PETSC_COMM_WORLD, 2, is_unionArray, &is_union);
    ISSort(is_union);
    PetscInt rstart, rend;
    MatGetOwnershipRange(jacPCMat, &rstart, &rend);
    ISComplement(is_union, rstart, rend, &is_flowState);

    // set IS to PC
    // Order must be surfaceState, flowState, modelState, this order will get best performance.
    PCFieldSplitSetIS(MLRGlobalPC, "surfaceState", is_surfaceState);
    PCFieldSplitSetIS(MLRGlobalPC, "flowState", is_flowState);
    PCFieldSplitSetIS(MLRGlobalPC, "modelState", is_modelState);

    PCFieldSplitSetType(MLRGlobalPC, PC_COMPOSITE_MULTIPLICATIVE);

    if (daOption_.getOption<label>("debug"))
    {
        KSPSetComputeEigenvalues(ksp, PETSC_TRUE);
    }

    //Setup the main ksp context before extracting the subdomains
    KSPSetUp(ksp);

    // Surface state uses identity; flow and model states use ASM(ILU).
    PetscInt nsub;
    KSP* subfieldksp;
    PCFieldSplitGetSubKSP(MLRGlobalPC, &nsub, &subfieldksp);

    MatOrderingType localMatrixOrdering;
    if (jacMatReOrdering == "natural")
    {
        localMatrixOrdering = MATORDERINGNATURAL;
    }
    else if (jacMatReOrdering == "nd")
    {
        localMatrixOrdering = MATORDERINGND;
    }
    else if (jacMatReOrdering == "rcm")
    {
        localMatrixOrdering = MATORDERINGRCM;
    }
    else if (jacMatReOrdering == "1wd")
    {
        localMatrixOrdering = MATORDERING1WD;
    }
    else if (jacMatReOrdering == "qmd")
    {
        localMatrixOrdering = MATORDERINGQMD;
    }
    else
    {
        Info << "matOrdering not known. Using default: nested dissection" << endl;
        localMatrixOrdering = MATORDERINGND;
    }

    for (PetscInt i = 0; i < nsub; i++)
    {
        PC subfieldpc;
        KSPSetType(subfieldksp[i], KSPPREONLY);
        KSPGetPC(subfieldksp[i], &subfieldpc);

        // The surface-state correction is intentionally an identity action.
        // Avoid creating ASM subdomains and submatrices for this large block.
        if (i == 0)
        {
            PCSetType(subfieldpc, PCNONE);
            continue;
        }

        PCSetType(subfieldpc, PCASM);
        // Overlap must be specified before PCSetUp constructs the subdomains.
        PCASMSetOverlap(subfieldpc, asmOverlap);
        PCSetUp(subfieldpc);

        KSP* subfieldkspsubdomain;
        PetscInt ndomain;
        PCASMGetSubKSP(subfieldpc, &ndomain, NULL, &subfieldkspsubdomain);
        for (PetscInt j = 0; j < ndomain; j++)
        {
            PC subfieldsubdomianpc;
            KSPSetType(subfieldkspsubdomain[j], KSPPREONLY);
            KSPGetPC(subfieldkspsubdomain[j], &subfieldsubdomianpc);
            PCSetType(subfieldsubdomianpc, PCILU);
            PCFactorSetPivotInBlocks(subfieldsubdomianpc, PETSC_TRUE);
            PCFactorSetShiftType(subfieldsubdomianpc, MAT_SHIFT_NONZERO);
            PCFactorSetShiftAmount(subfieldsubdomianpc, PETSC_DECIDE);
            PCFactorSetMatOrderingType(
                subfieldsubdomianpc, localMatrixOrdering);
            PCFactorSetLevels(subfieldsubdomianpc, pcFillLevel);
        }
    }

    // PCFieldSplitGetSubKSP allocates only the pointer array. The KSP objects
    // remain owned by PCFIELDSPLIT and must not be destroyed here.
    PetscFree(subfieldksp);

    // Set the norm to unpreconditioned
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    // Setup monitor if necessary:
    if (printInfo)
    {
        KSPMonitorSet(ksp, myKSPMonitor, this, 0);
    }

    PetscInt maxIts = gmresMaxIters;
    PetscScalar rtol, atol;
    assignValueCheckAD(rtol, gmresRelTol);
    assignValueCheckAD(atol, gmresAbsTol);
    KSPSetTolerances(ksp, rtol, atol, PETSC_DEFAULT, maxIts);

    if (printInfo)
    {
        Info << "Solver Type: " << kspObjectType << endl;
        Info << "GMRES Restart: " << restartGMRES << endl;
        Info << "ASM Overlap: " << asmOverlap << endl;
        Info << "Mat ReOrdering: " << jacMatReOrdering << endl;
        Info << "ILU PC Fill Level: " << pcFillLevel << endl;
        Info << "GMRES Max Iterations: " << maxIts << endl;
        Info << "GMRES Relative Tolerance: " << rtol << endl;
        Info << "GMRES Absolute Tolerance: " << atol << endl;
    }

    ISDestroy(&is_flowState);
    ISDestroy(&is_modelState);
    ISDestroy(&is_surfaceState);
    ISDestroy(&is_union);
}

} // End namespace Foam

// ************************************************************************* //
