/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for max von Mises stress

    Options template:
            "maxTKS" : {
            "type"         : "scalarFieldKS",
            "source"       : "allCells",
            "coeffKS"      : 50,
            "isGrad"       : False,
            "ksAggregationType": "continuous"/"discrete",
            "scale"        : 1.0}
            
\*---------------------------------------------------------------------------*/

#ifndef DAFunctionScalarFieldKS_H
#define DAFunctionScalarFieldKS_H

#include "DAFunction.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFunctionScalarFieldKS Declaration
\*---------------------------------------------------------------------------*/

class DAFunctionScalarFieldKS
    : public DAFunction
{

protected:
    /// coefficient for the KS function
    scalar coeffKS_;
    bool isGrad_;

    scalar normalValue_;
    word normalMode_;

    /// name of the variable
    word fieldName_;

    // ksAggregationType "continuous" or "discrete"
    word ksAggregationType_;

    /// log coeff
    scalar alpha_;

public:
    TypeName("scalarFieldKS");
    // Constructors

    //- Construct from components
    DAFunctionScalarFieldKS(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex,
        const word functionName);

    //- Destructor
    virtual ~DAFunctionScalarFieldKS()
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
