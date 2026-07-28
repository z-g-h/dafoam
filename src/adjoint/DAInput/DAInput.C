/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInput.H"
#include "Pstream.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAInput, 0);
defineRunTimeSelectionTable(DAInput, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInput::DAInput(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : inputName_(inputName),
      inputType_(inputType),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex)
{
    // initialize stateInfo_
    word solverName = daOption_.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAInput> DAInput::New(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting input: " << inputType << " for DAInput." << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(inputType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAInput::New"
            "("
            "    const word,"
            "    fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&"
            ")")
            << "Unknown DAInput type "
            << inputType << nl << nl
            << "Valid DAInput types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAInput>(
        cstrIter()(inputName,
                   inputType,
                   mesh,
                   daOption,
                   daModel,
                   daIndex));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

label DAInput::jacobianInputSize()
{
    return size();
}

label DAInput::jacobianRegisteredInputSize()
{
    return jacobianInputSize();
}

void DAInput::setJacobianInputValues(
    scalarList& jacobianInput,
    const double* physicalInput)
{
    forAll(jacobianInput, idxI)
    {
        jacobianInput[idxI] = physicalInput[idxI];
    }
}

void DAInput::runForJacobian(const scalarList& jacobianInput)
{
    run(jacobianInput);
}

void DAInput::getJacobianInputProduct(const scalarList& jacobianInput, double* product)
{
#ifdef CODI_ADR
    forAll(jacobianInput, idxI)
    {
        product[idxI] = jacobianInput[idxI].getGradient();

        // A serial input is replicated on every processor. Its reverse
        // contribution must therefore be summed and returned consistently on
        // every processor. Distributed inputs already own their local entries.
        if (!distributed())
        {
            reduce(product[idxI], sumOp<double>());
        }
    }
#else
    // These hooks are called only by the reverse-AD implementation of
    // DASolver::calcJacTVecProduct. Keep passive and forward builds
    // source-compatible.
    (void)jacobianInput;
    (void)product;
#endif
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
