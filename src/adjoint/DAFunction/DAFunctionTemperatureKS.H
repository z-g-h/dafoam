/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for max von Mises stress

    Options template:
            "maxTKS" : {
            "type"         : "temperatureKS",
            "source"       : "allCells",
            "coeffKS"      : 50,
            "isGrad"       : False,
            "scale"        : 1.0}
            
\*---------------------------------------------------------------------------*/

#ifndef DAFunctionTemperatureKS_H
#define DAFunctionTemperatureKS_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionTemperatureKS Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionTemperatureKS
    : public DAFunction
{

protected:
    /// coefficient for the KS function
    scalar coeffKS_;
    bool isGrad_;

public:
    TypeName("temperatureKS");
    // Constructors

    //- Construct from components
    DAFunctionTemperatureKS(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionTemperatureKS()
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
