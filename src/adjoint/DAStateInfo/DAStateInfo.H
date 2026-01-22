/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Register the names of state variables and how do they connect
        with their residuals

\*---------------------------------------------------------------------------*/

#ifndef DAStateInfo_H
#define DAStateInfo_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAStateInfo Declaration
\*---------------------------------------------------------------------------*/

class DAStateInfo
{

private:
    /// Disallow default bitwise copy construct
    DAStateInfo(const DAStateInfo&);

    /// Disallow default bitwise assignment
    void operator=(const DAStateInfo&);

protected:
    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// registered states 1st hash: solverName, 2nd hash: fieldType, 3nd list, stateNames
    HashTable<wordList> stateInfo_;

    /// restore model state
    wordList modelState_;

    /// table to specify how the states are connected to the residuals for a given solver
    HashTable<List<List<word>>> stateResConInfo_;

public:
    /// Runtime type information
    TypeName("DAStateInfo");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAStateInfo,
        dictionary,
        (const word modelType,
         const fvMesh& mesh,
         const DAOption& daOption,
         const DAModel& daModel),
        (modelType, mesh, daOption, daModel));

    // Constructors

    //- Construct from components
    DAStateInfo(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAStateInfo> New(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    //- Destructor
    virtual ~DAStateInfo()
    {
    }

    /// return the reference of stateInfo_
    const HashTable<wordList>& getStateInfo() const
    {
        return stateInfo_;
    }

    /// return the reference of stateResConInfo_
    const HashTable<List<List<word>>>& getStateResConInfo() const
    {
        return stateResConInfo_;
    }

    /// clear model state from stateInfo for frozenTurbulence mode.
    /// restore model state in modelState_.
    void clearModelStateInfo()
    {
        modelState_.swap(stateInfo_["modelStates"]);
    }

    /// clear model state from stateInfo for frozenTurbulence
    /// this function should be called before clearModelStateInfo()
    void clearModelStateResConInfo();
    void removeConModelState(List<List<word>>& stateCon);

    /// return model state size
    label getModelStateSize()
    {
        return stateInfo_["modelStates"].size() + modelState_.size();
    }

    /// return the reference modelStates
    const wordList& getModelStates() const
    {
        if (stateInfo_["modelStates"].size() != 0)
        {
            return stateInfo_["modelStates"];
        }
        else
        {
            return modelState_;
        }
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
