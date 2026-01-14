/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DALinearEqnFSturb.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DALinearEqnFSturb, 0);
addToRunTimeSelectionTable(DALinearEqn, DALinearEqnFSturb, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DALinearEqnFSturb::DALinearEqnFSturb(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAIndex& daIndex)
    : DALinearEqn(mesh, daOption, daIndex)
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DALinearEqnFSturb::createMLRKSP(
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

    PetscInt rstart, rend;
    MatGetOwnershipRange(jacPCMat, &rstart, &rend);

    label gmresRestart =
        daOption_.getSubDictOption<label>("adjEqnOption", "gmresRestart");
    label asmOverlap =
        daOption_.getSubDictOption<label>("adjEqnOption", "asmOverlap");
    label globalPCIters =
        daOption_.getSubDictOption<label>("adjEqnOption", "globalPCIters");
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
    //               |            Usually Additive Schwartz and overlap is set
    //               |            with 'ASMOverlap'. Use 0 to get BlockJacobi
    //               |
    //               --> subKSP --> KSP type set to Richardon with 'LocalPreConIts'
    //                   |
    //                   --> subPC -->  PC type set to 'localPCType'.
    //                                  Usually ILU. 'localFillLevel' is
    //                                  set and 'localMatrixOrder' is used.
    //
    // Note that if globalPreConIts=1 then maser_PC_KSP is NOT created and master_PC=globalPC
    // and if localPreConIts=1 then subKSP is set to preOnly.

    // First, KSPSetFromOptions MUST be called
    PetscOptionsSetValue(NULL, "-ksp_view_eigenvalues", NULL);
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

    // Set global and local PC iters
    PetscInt globalPreConIts = globalPCIters;

    // Since there is an extraneous matMult required when using the
    // richardson precondtiter with only 1 iteration, only use it when we need
    // to do more than 1 iteration.
    // Just pull out the pc-object if we are not using kspRichardson
    KSPGetPC(ksp, &MLRGlobalPC);

    // Set the type of 'MLRGlobalPC'. This will almost always be additive schwartz
    PCSetType(MLRGlobalPC, PCFIELDSPLIT);

    IS is_flow, is_turb;
    label nLocalCells = daIndex_.nLocalCells;
    PetscInt turbState_indices[2 * nLocalCells];
    forAll(mesh_.cells(), cellI)
    {
        PetscInt omegaGlobalIdx = daIndex_.getGlobalAdjointStateIndex("omega", cellI);
        turbState_indices[2* cellI] = omegaGlobalIdx;
        PetscInt kGlobalIdx = daIndex_.getGlobalAdjointStateIndex("k", cellI);        
        turbState_indices[2*cellI + 1] = kGlobalIdx;
    }
    ISCreateGeneral(PETSC_COMM_WORLD, 2 * nLocalCells, turbState_indices,  PETSC_COPY_VALUES, &is_turb);
    ISSort(is_turb);

    ISComplement(is_turb, rstart, rend, &is_flow);

    List<word> splitNames = {"turb", "flow"};
    PCFieldSplitSetIS(MLRGlobalPC, "turb", is_turb);
    PCFieldSplitSetIS(MLRGlobalPC, "flow", is_flow);

    PCFieldSplitSetType(MLRGlobalPC, PC_COMPOSITE_ADDITIVE);
    // Set the overlap required



    //label KSPCalcEigen = readLabel(options.lookup("KSPCalcEigen"));
    //if (KSPCalcEigen)
    //{
    KSPSetComputeEigenvalues(ksp, PETSC_TRUE);
    //}

    //Setup the main ksp context before extracting the subdomains
    KSPSetUp(ksp);
    PetscInt nsub;
    word matOrdering = jacMatReOrdering;
    PetscInt localFillLevel = pcFillLevel;
    PetscInt asmoverlap = asmOverlap;
    KSP* subfieldksp;
    PCFieldSplitGetSubKSP(MLRGlobalPC, &nsub, &subfieldksp);
    for (PetscInt i = 0; i < nsub; i++)
    {
        PC subfieldpc;
        KSPGetPC(subfieldksp[i], &subfieldpc);
        PCSetType(subfieldpc, PCASM);
        PCSetUp(subfieldpc);
        PCASMSetOverlap(subfieldpc, asmoverlap);
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
            // PCFactorSetDropTolerance(MLRsubpc, localDropTol, localDropTcol, localDropMaxRowCount);
            // PCFactorSetAllowDiagonalFill(MLRsubpc, PETSC_TRUE);

            // Setup the matrix ordering for the subpc object:
            // 'natural':'natural',
            // 'rcm':'rcm',
            // 'nested dissection':'nd' (default),
            // 'one way dissection':'1wd',
            // 'quotient minimum degree':'qmd',
            MatOrderingType localMatrixOrdering;
            if (matOrdering == "natural")
            {
                localMatrixOrdering = MATORDERINGNATURAL;
            }
            else if (matOrdering == "nd")
            {
                localMatrixOrdering = MATORDERINGND;
            }
            else if (matOrdering == "rcm")
            {
                localMatrixOrdering = MATORDERINGRCM;
            }
            else if (matOrdering == "1wd")
            {
                localMatrixOrdering = MATORDERING1WD;
            }
            else if (matOrdering == "qmd")
            {
                localMatrixOrdering = MATORDERINGQMD;
            }
            else if (matOrdering == "amd")
            {
                localMatrixOrdering = MATORDERINGAMD;
            }
            else if (matOrdering == "metisnd")
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
            PCFactorSetLevels(subfieldsubdomianpc, localFillLevel);
        }
    }

    KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);

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
        // Info << "ASM Overlap: " << MLRoverlap << endl;
        Info << "Global PC Iters: " << globalPreConIts << endl;
        Info << "GMRES Max Iterations: " << maxIts << endl;
        Info << "GMRES Relative Tolerance: " << rtol << endl;
        Info << "GMRES Absolute Tolerance: " << atol << endl;
    }
}

} // End namespace Foam

// ************************************************************************* //
