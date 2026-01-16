/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionEntropyGeneration.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionEntropyGeneration, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionEntropyGeneration, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionEntropyGeneration::DAFunctionEntropyGeneration(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName)
{
    // read entropy changes options
    if (functionDict_.found("inletPatches") && functionDict_.found("outletPatches"))
    {
        // read patches names
        functionDict_.readEntry<wordList>("inletPatches", inletPatches_);
        functionDict_.readEntry<wordList>("outletPatches", outletPatches_);

        // read reference temperature and pressure
        functionDict_.readEntry<scalar>("referencePressure", PRef_);
        functionDict_.readEntry<scalar>("referenceTemperature", TRef_);   
    }
    // read entropy transfers options
    if (functionDict_.found("heatExchangePatches"))
    {
        functionDict_.readEntry<wordList>("heatExchangePatches", heatExchangePatches_);
    }

    // get Cp from thermophysicalProperties
    const IOdictionary& thermoDict = mesh_.thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
    dictionary mixSubDict = thermoDict.subDict("mixture");
    dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
    Cp_ = thermodynamicsSubDict.getScalar("Cp");
    gamma_ = thermodynamicsSubDict.getScalar("gamma");

    // debug
    if (daOption_.getOption<label>("debug"))
    {
        Info << "Cp " << Cp_ << endl;
        Info << "gamma " << gamma_ << endl;
    }
}

// calculate the value of objective function
scalar DAFunctionEntropyGeneration::calcFunction()
{
    /*
    Description:
        calclute the steady entropy generation
        entropy changes = entropy transfers + entropy generation

        entropy changes = entropyFlow(outlet) - entropyFlow(inlet)
        entropyFlow = massflow * specificEntropy

        entropy trnsfers = ∫ ( q / T ) dA, mainly caused by wall heat tranfers.

        for compresssible flow, specific entropy expression as follows:
        s = Cp * ln(T/TRef) - R * ln(p/PRef)
        R = Cp - Cp/γ
    */

    const objectRegistry& db = mesh_.thisDb();
    const volVectorField& U = db.lookupObject<volVectorField>("U");
    const volScalarField& rho = db.lookupObject<volScalarField>("rho");
    const volVectorField::Boundary& UBf = U.boundaryField();
    const volScalarField::Boundary& rhoBf = rho.boundaryField();

    // initialization function values, get field variables
    scalar functionValue = 0.0;

    DATurbulenceModel& daTurbModel =
             const_cast<DATurbulenceModel&>(daModel_.getDATurbulenceModel());

    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    fluidThermo& thermo = const_cast<fluidThermo&>(
                    mesh_.thisDb().lookupObject<fluidThermo>("thermophysicalProperties"));
    volScalarField& he = thermo.he();
    volScalarField alphaEff = daTurbModel.alphaEff();
    const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();
  
    const scalar R = Cp_ - Cp_ / gamma_;
    const scalar expCoeff = gamma_ / (gamma_ - 1.0);

    const volScalarField::Boundary& pBf = p.boundaryField();
    const volScalarField::Boundary& TBf = T.boundaryField();
    const volScalarField::Boundary& heBf = he.boundaryField();
    const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();

    // calculate inlet and outlet entropy changes
    scalar entropyFlowInlet = 0.0;
    scalar entropyFlowOutlet = 0.0;
    scalar totalEntropyChanges = 0;
    scalar totalEntropyTransfers = 0;

    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        word patchName = mesh_.boundaryMesh()[patchI].name();

        if (inletPatches_.found(patchName) || outletPatches_.found(patchName))
        {
            // calculate massflow rate
            vector US = UBf[patchI][faceI];
            vector Sf = mesh_.Sf().boundaryField()[patchI][faceI];
            scalar rhoS = rhoBf[patchI][faceI];
            scalar mfr = rhoS * (US & Sf);  

            // calculate specific entropy 
            scalar TS = TBf[patchI][faceI];
            scalar pS = pBf[patchI][faceI];
            scalar UMag = mag(UBf[patchI][faceI]);
            scalar Ma2 = sqr(UMag / sqrt(gamma_ * R * TS));
            scalar TT = TS * (1.0 + 0.5 * (gamma_ - 1.0) * Ma2);
            scalar pT = pS * pow(1.0 + 0.5 * (gamma_ - 1.0) * Ma2, expCoeff);    

            scalar specificEntropy = Cp_ * log(TT / TRef_) - R * log(pT / PRef_); 

            // calculate entropy by massflowRate * specificEntropy
            scalar entropyFlow = mfr * specificEntropy;   

            // sum the entropy
            if (inletPatches_.found(patchName))
            {
                entropyFlowInlet += entropyFlow;
            }
            else if (outletPatches_.found(patchName))
            {
                entropyFlowOutlet += entropyFlow;
            }
        }
        if (heatExchangePatches_.found(patchName))
        {
            scalar magSf = mesh_.magSf().boundaryField()[patchI][faceI];
            scalar TS = TBf[patchI][faceI];
            scalar dHedz = heBf[patchI].snGrad()()[faceI]; 

            // calculate heat flux (heat flux = alphaEff* dHe/dz)
            scalar qWall = alphaEffBf[patchI][faceI] * dHedz;
        
            // d S_transfers = (q_wall / TS) * dA
            scalar entropyTransfer = (qWall * magSf) / TS;

            // scalar entropyGen = qWall * area / Twall;
            totalEntropyTransfers += entropyTransfer;
        }

    }

    reduce(entropyFlowInlet, sumOp<scalar>());
    reduce(entropyFlowOutlet, sumOp<scalar>());
    reduce(totalEntropyTransfers, sumOp<scalar>());

    // calculate entropy generation by entropy balance equation
    totalEntropyChanges = entropyFlowOutlet + entropyFlowInlet;
    functionValue = totalEntropyChanges - totalEntropyTransfers;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //




