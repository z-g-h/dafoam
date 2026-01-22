/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAStateInfo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAStateInfo, 0);
defineRunTimeSelectionTable(DAStateInfo, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAStateInfo::DAStateInfo(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel)
{

    // initialize stateInfo
    stateInfo_.set("volScalarStates", {});
    stateInfo_.set("volVectorStates", {});
    stateInfo_.set("surfaceScalarStates", {});
    stateInfo_.set("modelStates", {});

    //Info<<stateInfo_<<endl;
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAStateInfo> DAStateInfo::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DAStateInfo" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAStateInfo::New"
            "("
            "    const word,"
            "    const fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&"
            ")")
            << "Unknown DAStateInfo type "
            << modelType << nl << nl
            << "Valid DAStateInfo types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    autoPtr<DAStateInfo> daStateInfoPtr = autoPtr<DAStateInfo>(
        cstrIter()(modelType, mesh, daOption, daModel));
    
    if (daOption.getOption<label>("frozenTurbulence"))
    {
        daStateInfoPtr->clearModelStateResConInfo();
        daStateInfoPtr->clearModelStateInfo();
    }
    return daStateInfoPtr;
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAStateInfo::clearModelStateResConInfo()
{
    forAll(stateInfo_["volVectorStates"], idxI)
    {
        word stateName = stateInfo_["volVectorStates"][idxI];
        word resName = stateName + "Res";
        this->removeConModelState(stateResConInfo_[resName]);
    }
    forAll(stateInfo_["volScalarStates"], idxI)
    {
        word stateName = stateInfo_["volScalarStates"][idxI];
        word resName = stateName + "Res";
        this->removeConModelState(stateResConInfo_[resName]);
    }
    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        word stateName = stateInfo_["surfaceScalarStates"][idxI];
        word resName = stateName + "Res";
        this->removeConModelState(stateResConInfo_[resName]);
    }
}

void DAStateInfo::removeConModelState(List<List<word>>& stateCon)
{
    forAll(stateCon, idxI)
    {
        List<word> tmpStateCon;
        forAll(stateCon[idxI], idxJ)
        {
            word conStateName = stateCon[idxI][idxJ];
            // only conStateName not be modelStates, we store it.
            if (!(stateInfo_["modelStates"].found(conStateName) || modelState_.found(conStateName)))
            {
                tmpStateCon.append(conStateName);
            }
        }
        stateCon[idxI].swap(tmpStateCon);
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
