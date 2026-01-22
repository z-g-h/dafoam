/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for gas entropy generation calculation, only support compressible flow.
 
    Options template:
            "entropyGen": {
                            "type": "entropyGeneration",
                            "source": "patchToFace",
                            "patches": ["inlet", "outlet", "wall"],
        entropyChanges
        options        |    "inletPatches": ["inlet"],
                    |    "outletPatches":["outlet"],
                    |    "referencePressure": 1e5,
                    |    "referenceTemperature": 300,
        entropyTransfers
        options        |    "heatExchangePatches": ["wall"],

                        "scale": 1.0,
                        "addToAdjoint": True},
        
        if entropy generation only has entropyChanges, there is no need to set entropy transfers options.
    
\*---------------------------------------------------------------------------*/

#ifndef DAFunctionEntropyGeneration_H
#define DAFunctionEntropyGeneration_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionEntropyGeneration Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionEntropyGeneration
    : public DAFunction
{

protected:
    /// names of outlet patches
    wordList outletPatches_;

    /// names of inlet patches
    wordList inletPatches_;

    /// name of heat exchange patches
    wordList heatExchangePatches_;

    scalar Cp_;

    scalar gamma_;

    /// reference
    scalar PRef_;
    scalar TRef_;

    /// total

public:
    TypeName("entropyGeneration");
    // Constructors

    //- Construct from components
    DAFunctionEntropyGeneration(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionEntropyGeneration()
    {
    }

    /// calculate the value of objective function
    virtual scalar calcFunction();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
