/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Compute Jacobian connectivity and coloring

\*---------------------------------------------------------------------------*/

#ifndef DAJacCon_H
#define DAJacCon_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAUtility.H"
#include "DAOption.H"
#include "DAIndex.H"
#include "DAModel.H"
#include "DAStateInfo.H"
#include "syncTools.H"
#include "DAFunction.H"
#include "DAColoring.H"
#include "DAField.H"
#include "labelList.H"
#include "mixingPlanePolyPatch.H"
#include <unordered_map>
#include <utility> 

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAJacCon Declaration
\*---------------------------------------------------------------------------*/

class DAJacCon
{

private:
    /// Disallow default bitwise copy construct
    DAJacCon(const DAJacCon&);

    /// Disallow default bitwise assignment
    void operator=(const DAJacCon&);

protected:
    /// the name of the jacCon matrix
    const word modelType_;

    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// DAColoring object
    DAColoring daColoring_;

    /// DAField object
    DAField daField_;

    /// the regState_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

    /// matrix to store boundary connectivity levels for state Jacobians
    Mat stateBoundaryCon_;
    /// matrix to store field boundary connectivity levels for state Jacobians
    Mat fieldStateBoundaryCon_;

    /// matrix to store boundary connectivity ID for state Jacobians
    Mat stateBoundaryConID_;
    /// matrix to store field boundary connectivity ID for state Jacobians
    Mat fieldStateBoundaryConID_;

    /// neibough face global index for a given local boundary face
    ///     for coupled boundary    
    labelList neiBFaceGlobalCompact_;
    ///     for field coupled boundary e.g. cyclicAMI, mixingPlane
    labelListList fieldNeiBFaceGlobalCompact_;

    /// Jacobian connectivity mat
    Mat jacCon_;

    /// jacCon matrix colors
    Vec jacConColors_;

    /// number of jacCon colors
    label nJacConColors_;

    /// \name on- and off-diagonal preallocation sizes for each row of dRdW, dRdWT, and dRdXv
    //@{
    Vec dRdWTPreallocOn_;
    Vec dRdWTPreallocOff_;
    Vec dRdWPreallocOn_;
    Vec dRdWPreallocOff_;
    //@}

    /// a map to temperary store the connectivity, used to setup jacCon_ and boundaryCon_
    typedef std::unordered_map<PetscInt, std::unordered_map<PetscInt, PetscScalar>> bufferConMap;

    /// initialize state boundary connection
    void initializeStateBoundaryCon();

    /// calculate DAJacCon::neiBFaceGlobalCompact_
    void calcNeiBFaceGlobalCompact(labelList& neiBFaceGlobalCompact);
    /// calculate DAJacCon::fieldNeiBFaceGlobalCompact_
    void calcFieldNeiBFaceGlobalCompact(labelListList& fieldNeiBFaceGlobalCompact);

    /// given a local face index, return the local index of the coupled boundary face
    label getLocalCoupledBFaceIndex(const label localFaceI) const;
    /// given a lolcal face index, reuturn the local index of the field coupled boundary face
    label getLocalFieldCoupledBFaceIndex(const label localFaceI) const;

    /// calculate DAJacCon::stateBoundaryCon_
    void setupStateBoundaryCon(Mat* stateBoundaryCon, Mat* fieldStateBoundaryCon);

    /// calculate DAJacCon::stateBoundaryConID_
    void setupStateBoundaryConID(Mat* stateBoundaryConID, Mat* fieldStateBoundaryConID);

    /// a high-level function to add connected state column indices to the connectivity matrix
    void addStateConnections(
        bufferConMap &connections,
        const label cellI,
        const label connectedLevelLocal,
        const wordList connectedStatesLocal,
        const List<List<word>> connectedStateInterProc,
        const label addFace);

    /// add value 1 for the colume idx to conMat
    void setConnections(
        bufferConMap &conMat,
        const label idx) const;

    /// assign values in connections to a specific row idxI in conMat
    void setupJacobianConnections(
        Mat conMat,
        bufferConMap &connections,
        const label idxI);

    /// combine stateBoundaryCon and stateBoundaryConTmp, this is to ensure including all connected states for parallel cases
    void combineStateBndCon(
        Mat* stateBoundaryCon,
        bufferConMap &stateCon_map,
        Mat* fieldStateBoundaryCon,
        bufferConMap &fieldStateCon_map);
    
    /// preallocate stateBoundaryCon and fieldStateBoundaryCon
    void preallocateStateBndCon(
        PetscInt localrows,
        Mat* bndConMat,
        bufferConMap &con_map);

    /// add val to the gRow row in conMat, the column indice are the state (given by stateName) for a given cellI
    void addConMatCell(
        bufferConMap &conMat,
        const label gRow,
        const label cellI,
        const word stateName,
        const PetscScalar val);

    /// overload function for cyclicAMI function
    void addConMatCell(
        bufferConMap &conMat,
        const labelList gRows,
        const label cellI,
        const word stateName,
        const PetscScalar val);

    /// add val to gRow row in conMat, column indice are the neighbouring states (given by stateName) for a given cellI
    void addConMatNeighbourCells(
        bufferConMap &conMat,
        const label gRow,
        const label cellI,
        const word stateName,
        const PetscScalar val);

    /// overload function for cyclicAMI function
    void addConMatNeighbourCells(
        bufferConMap &conMat,
        const labelList gRows,
        const label cellI,
        const word stateName,
        const PetscScalar val);    

    /// add val to gRow row in conMat, column indice are the faces (given by stateName) for a given cellI
    void addConMatCellFaces(
        bufferConMap &conMat,
        const label gRow,
        const label cellI,
        const word stateName,
        const PetscScalar val);

    /// overload function for cyclicAMI function
    void addConMatCellFaces(
        bufferConMap &conMat,
        const labelList gRows,
        const label cellI,
        const word stateName,
        const PetscScalar val);    

    /// add the column index of the (iner-proc) connected states and faces to conMat, given a local face index
    void addBoundaryFaceConnections(
        bufferConMap &conMat,
        const label gRow,
        const label cellI,
        const labelList v,
        const List<List<word>> connectedStates,
        const label addFaces);

    /// overload function for cyclicAMI and mixingPlane only used in setupStateBoundaryCon function
    void addBoundaryFaceConnections(
        bufferConMap &conMat,
        const labelList gRows,
        const label cellI,
        const labelList v,
        const List<List<word>> connectedStates,
        const label addFaces);   

    /// clear members in DAJacCon
    void clearDAJacConMembers()
    {
        // Need to clean up all the matrices
        MatDestroy(&jacCon_);
        MatDestroy(&stateBoundaryCon_);
        MatDestroy(&fieldStateBoundaryCon_);        
        MatDestroy(&stateBoundaryConID_);
        MatDestroy(&fieldStateBoundaryConID_);
        VecDestroy(&jacConColors_);
        stateInfo_.clear();
        neiBFaceGlobalCompact_.clear();
        fieldNeiBFaceGlobalCompact_.clear();
    }

    /// check if there is special boundary conditions that need special treatment in jacCon_
    void checkSpecialBCs();

    /// a vector to show whether a state is connected to a pressureInletVelocity boundary face (3 level max)
    Vec isPIVBCState_;

    /// function used to add connectivity for pressureInletVelocity
    void setPIVVec(
        Vec iSPIV,
        const label idxI);

    /// add connectivity phi for pressureInletVelocity
    label addPhi4PIV(
        const word stateName,
        const label idxI,
        const label comp = -1);

    /// initialize petsc vectors
    void initializePetscVecs();

    /// compute preallocation vectors
    void allocateJacobianConnections(
        Vec preallocOnProc,
        Vec preallocOffProc,
        Vec preallocOnProcT,
        Vec preallocOffProcT,
        bufferConMap &connections,
        const label row);

    /// compute preallocation vectors
    void preallocateJacobianMatrix(
        Mat dRMat,
        const Vec preallocOnProc,
        const Vec preallocOffProc) const;

    void setupdRdWCon(
        const HashTable<List<List<word>>>& stateResConInfo,
        const label isPrealloc);

public:
    // Constructors
    DAJacCon(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAJacCon()
    {
    }

    // Member functions

    /// clear members in parent and child objects
    void clear();

    /// calculate the preallocation vector for initializing the JacCon mat, if necessary
    void setupJacConPreallocation(const dictionary& options);

    /// initialize the state Jacobian connectivity matrix
    void initializeJacCon(const dictionary& options);

    /// assign 1 to all non-zero elements for the Jacobian connecitivyt matrix
    void setupJacCon(const dictionary& options);

    /// get the number of JacCon colors
    label getNJacConColors() const
    {
        return nJacConColors_;
    }

    /// preallocate dRdW matrix using the preallocVec
    void preallocatedRdW(
        Mat dRMat,
        const label transposed) const;

    /// compute graph coloring for Jacobian connectivity matrix
    void calcJacConColoring(const word postFix = "");

    /// read colors for JacCon
    void readJacConColoring(const word postFix = "");

    /// whether the coloring file exists
    label coloringExists(const word postFix = "") const;

    /// return DAJacCon::jacConColors_
    Vec getJacConColor() const
    {
        return jacConColors_;
    }

    /// calculate the colored column vector
    void calcColoredColumns(
        const label colorI,
        Vec coloredColumn) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
