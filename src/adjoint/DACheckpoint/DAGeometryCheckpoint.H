/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Geometry interface used to split reverse-mode volume-coordinate
        products into mesh-geometry and residual/function tape segments.

\*---------------------------------------------------------------------------*/

#ifndef DAGeometryCheckpoint_H
#define DAGeometryCheckpoint_H

#include "fvCFD.H"

namespace Foam
{

#ifdef CODI_ADR

class DAGeometryCheckpoint
{
private:
    struct AMIAddressingSnapshot
    {
        word patchName;
        label singlePatchProc;
        labelListList srcAddress;
        labelListList tgtAddress;

        AMIAddressingSnapshot()
            : patchName(word::null),
              singlePatchProc(-2)
        {
        }
    };

    fvMesh& mesh_;
    label referenceSize_;
    List<AMIAddressingSnapshot> amiAddressingSnapshots_;

    void captureAMIAddressing();

    void verifyAMIAddressing() const;

public:
    explicit DAGeometryCheckpoint(fvMesh& mesh);

    /// Capture the interface size and fixed AMI addressing after the passive
    /// mesh update that establishes the reference geometry.
    void captureReference();

    label interfaceSize() const
    {
        return referenceSize_;
    }

    /// Register the geometry interface as independent tape inputs for the
    /// residual/function segment.
    void registerInputs(codi::RealReverse::Tape& tape);

    /// Register the geometry interface as tape outputs for the mesh segment.
    void registerOutputs(codi::RealReverse::Tape& tape);

    void readGradients(List<double>& gradients);

    void seedGradients(const List<double>& gradients);

    void deactivateValues(codi::RealReverse::Tape& tape);

    /// Verify that replay rebuilt the same interface and did not change the
    /// discrete AMI interpolation addressing.
    void verifyReplay() const;
};

#endif

} // End namespace Foam

#endif

