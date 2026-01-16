/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DALinearEqnFSModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DALinearEqnFSModel, 0);
addToRunTimeSelectionTable(DALinearEqn, DALinearEqnFSModel, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DALinearEqnFSModel::DALinearEqnFSModel(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAIndex& daIndex)
    : DALinearEqn(mesh, daOption, daIndex)
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DALinearEqnFSModel::createMLRKSP(
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

    label gmresRestart =
        daOption_.getSubDictOption<label>("adjEqnOption", "gmresRestart");
    label asmOverlap =
        daOption_.getSubDictOption<label>("adjEqnOption", "asmOverlap");
    word jacMatReOrdering =
        daOption_.getSubDictOption<word>("adjEqnOption", "jacMatReOrdering");
    label pcFillLevel =
        daOption_.getSubDictOption<label>("adjEqnOption", "pcFillLevel");
    label gmresMaxIters =
        daOption_.getSubDictOption<label>("adjEqnOption", "gmresMaxIters");
    scalar gmresRelTol =
        daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresRelTol");
    scalar gmresAbsTol =
        daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresAbsTol");
    label useNonZeroInitGuess =
        daOption_.getSubDictOption<label>("adjEqnOption", "useNonZeroInitGuess");
    label useMGSO =
        daOption_.getSubDictOption<label>("adjEqnOption", "useMGSO");
    label printInfo =
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
    forAll(daIndex_.adjStateNames, idx)
    {
        word stateName = daIndex_.adjStateNames[idx];
        if (daIndex_.adjStateType[stateName] == "modelState")
        {
            modelStateSize++;
            modelState.append(stateName);
        }
    }
    
    // exact model state IS
    IS is_flow, is_model;
    label nLocalCells = daIndex_.nLocalCells;
    PetscInt modelState_indices[modelStateSize * nLocalCells];
    forAll(mesh_.cells(), cellI)
    {
        for (PetscInt idx = 0; idx < modelStateSize; idx++)
        {
            PetscInt globalIdx = daIndex_.getGlobalAdjointStateIndex(modelState[idx], cellI);
            modelState_indices[modelStateSize * cellI + idx] = globalIdx;
        }
    }
    ISCreateGeneral(PETSC_COMM_WORLD, modelStateSize * nLocalCells, modelState_indices, PETSC_COPY_VALUES, &is_model);
    ISSort(is_model);

    // use a complement to get flow state and surfaceState IS
    PetscInt rstart, rend;
    MatGetOwnershipRange(jacPCMat, &rstart, &rend);
    ISComplement(is_model, rstart, rend, &is_flow);

    // set IS to PC
    PCFieldSplitSetIS(MLRGlobalPC, "flow", is_flow);
    PCFieldSplitSetIS(MLRGlobalPC, "model", is_model);

    PCFieldSplitSetType(MLRGlobalPC, PC_COMPOSITE_MULTIPLICATIVE);

    //label KSPCalcEigen = readLabel(options.lookup("KSPCalcEigen"));
    //if (KSPCalcEigen)
    //{
    // KSPSetComputeEigenvalues(ksp, PETSC_TRUE);
    //}

    //Setup the main ksp context before extracting the subdomains
    KSPSetUp(ksp);

    // set every subfieldPC to ASM(ILU)
    PetscInt nsub;
    KSP* subfieldksp;
    PCFieldSplitGetSubKSP(MLRGlobalPC, &nsub, &subfieldksp);
    for (PetscInt i = 0; i < nsub; i++)
    {
        PC subfieldpc;
        KSPGetPC(subfieldksp[i], &subfieldpc);
        PCSetType(subfieldpc, PCASM);
        PCSetUp(subfieldpc);
        PCASMSetOverlap(subfieldpc, asmOverlap);
        KSP* subfieldkspsubdomain;
        PetscInt firstsub;
        PetscInt ndomain;
        PCASMGetSubKSP(subfieldpc, &ndomain, &firstsub, &subfieldkspsubdomain);
        for (PetscInt j = 0; j < ndomain; j++)
        {
            PC subfieldsubdomianpc;
            KSPSetType(subfieldkspsubdomain[j], KSPPREONLY);
            KSPGetPC(subfieldkspsubdomain[j], &subfieldsubdomianpc);
            PCSetType(subfieldsubdomianpc, PCILU);
            PCFactorSetPivotInBlocks(subfieldsubdomianpc, PETSC_TRUE);
            PCFactorSetShiftType(subfieldsubdomianpc, MAT_SHIFT_NONZERO);
            PCFactorSetShiftAmount(subfieldsubdomianpc, PETSC_DECIDE);

            // Setup the matrix ordering for the subpc object:
            // 'natural':'natural',
            // 'rcm':'rcm',
            // 'nested dissection':'nd' (default),
            // 'one way dissection':'1wd',
            // 'quotient minimum degree':'qmd',
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
            else if (jacMatReOrdering == "amd")
            {
                localMatrixOrdering = MATORDERINGAMD;
            }
            else if (jacMatReOrdering == "metisnd")
            {
                localMatrixOrdering = MATORDERINGMETISND;
            }
            else
            {
                Info << "matOrdering not known. Using default: nested dissection" << endl;
                localMatrixOrdering = MATORDERINGND;
            }
            PCFactorSetMatOrderingType(subfieldsubdomianpc, localMatrixOrdering);

            // Set the ILU parameters
            PCFactorSetLevels(subfieldsubdomianpc, pcFillLevel);
        }
    }

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
}

} // End namespace Foam

// ************************************************************************* //
