// bc/bc.d
// Base class for boundary condition objects, for use in Eilmer4
//
// Peter J. 2014-07-20 : first cut.
// RG & PJ  2015-12-03 : Decompose boundary conditions into lists of actions
//    

module bc;

import std.conv;
import std.json;
import std.stdio;
import std.string;
import gas;
import json_helper;
import geom;
import sgrid;
import fvcore;
import globalconfig;
import globaldata;
import flowstate;
import fvinterface;
import fvcell;
import block;
import sblock;
import fluxcalc;
import ghost_cell_effect;
import boundary_interface_effect;
import boundary_flux_effect;
import user_defined_effects;

BoundaryCondition make_BC_from_json(JSONValue jsonData, int blk_id, int boundary)
{
    auto newBC = new BoundaryCondition(blk_id, boundary);
    newBC.label = to!string(jsonData["label"]);
    newBC.type = to!string(jsonData["type"]);
    newBC.group = to!string(jsonData["group"]);
    newBC.is_wall = getJSONbool(jsonData, "is_wall", true);
    newBC.ghost_cell_data_available = getJSONbool(jsonData, "ghost_cell_data_available", true);
    newBC.convective_flux_computed_in_bc = getJSONbool(jsonData, "convective_flux_computed_in_bc", false);
    // Assemble list of preReconAction effects
    auto preReconActionList = jsonData["pre_recon_action"].array;
    foreach ( jsonObj; preReconActionList ) {
	newBC.preReconAction ~= make_GCE_from_json(jsonObj, blk_id, boundary);
    }
    auto postConvFluxActionList = jsonData["post_conv_flux_action"].array;
    foreach ( jsonObj; postConvFluxActionList ) {
	newBC.postConvFluxAction ~= make_BFE_from_json(jsonObj, blk_id, boundary);
    }
    auto preSpatialDerivActionList = jsonData["pre_spatial_deriv_action"].array;
    foreach ( jsonObj; preSpatialDerivActionList ) {
	newBC.preSpatialDerivAction ~= make_BIE_from_json(jsonObj, blk_id, boundary);
    }
    auto postDiffFluxActionList = jsonData["post_diff_flux_action"].array;
    foreach ( jsonObj; postDiffFluxActionList ) {
	newBC.postDiffFluxAction ~= make_BFE_from_json(jsonObj, blk_id, boundary);
    }
    // [TODO] Only need to the post convective flux option now.
    return newBC;
} // end make_BC_from_json()


class BoundaryCondition {
    // Boundary condition is built from composable pieces.
public:
    // Location of the boundary condition.
    Block blk; // the block to which this BC is applied
    int which_boundary; // identity/index of the relevant boundary
    // We may have a label for this specific boundary.
    string label;
    // We have a symbolic name for the type of boundary condition
    // when thinking about the flow problem conceptually. 
    string type;
    // Sometimes it is convenient to think of individual boundaries
    // grouped together.
    string group;
    // Nature of the boundary condition that may be checked 
    // by other parts of the CFD code.
    bool is_wall = true;
    bool ghost_cell_data_available = true;
    bool convective_flux_computed_in_bc = false;
    double emissivity = 0.0;
    FVInterface[] faces;
    BasicCell[] ghostcells;
    int[] outsigns;

private:
    // Working storage for boundary flux derivatives
    FlowState _Lft, _Rght;

public:
    this(int id, int boundary, bool isWall=true, bool ghostCellDataAvailable=true, double _emissivity=0.0)
    {
	blk = gasBlocks[id];  // pick the relevant block out of the collection
	which_boundary = boundary;
	type = "";
	group = "";
	is_wall = isWall;
	ghost_cell_data_available = ghostCellDataAvailable;
	emissivity = _emissivity;
	auto gm = GlobalConfig.gmodel_master;
	_Lft = new FlowState(gm);
	_Rght = new FlowState(gm);
    }

    // Action lists.
    // The BoundaryCondition is called at four stages in a global timestep.
    // Those stages are:
    // 1. pre reconstruction
    // 2. post convective flux evaluation
    // 3. pre spatial derivative estimate
    // 4. post diffusive flux evaluation
    // Note the object may be called more than 4 times depending
    // on the type of time-stepping used to advance the solution.
    // At each of these stages, a series of effects are applied in order
    // with the end goal to leave the boundary values in an appropriate
    // state. We will call this series of effects an action.
    GhostCellEffect[] preReconAction;
    BoundaryFluxEffect[] postConvFluxAction;
    BoundaryInterfaceEffect[] preSpatialDerivAction;
    BoundaryFluxEffect[] postDiffFluxAction;

    override string toString() const
    {
	char[] repr;
	repr ~= "BoundaryCondition(";
	repr ~= "label= " ~ label ~ ", type= " ~ type ~ ", group= " ~ group;
	repr ~= ", is_wall= " ~ to!string(is_wall);
	repr ~= ", ghost_cell_data_available= " ~ to!string(ghost_cell_data_available);
	repr ~= ", convective_flux_computed_in_bc= " ~ to!string(convective_flux_computed_in_bc);
	if ( preReconAction.length > 0 ) {
	    repr ~= ", preReconAction=[" ~ to!string(preReconAction[0]);
	    foreach (i; 1 .. preReconAction.length) {
		repr ~= ", " ~ to!string(preReconAction[i]);
	    }
	    repr ~= "]";
	}
	if ( postConvFluxAction.length > 0 ) {
	    repr ~= ", postConvFluxAction=[" ~ to!string(postConvFluxAction[0]);
	    foreach (i; 1 .. postConvFluxAction.length) {
		repr ~= ", " ~ to!string(postConvFluxAction[i]);
	    }
	    repr ~= "]";
	}
	if ( preSpatialDerivAction.length > 0 ) {
	    repr ~= ", preSpatialDerivAction=[" ~ to!string(preSpatialDerivAction[0]);
	    foreach (i; 1 .. preSpatialDerivAction.length) {
		repr ~= ", " ~ to!string(preSpatialDerivAction[i]);
	    }
	    repr ~= "]";
	}
	if ( postDiffFluxAction.length > 0 ) {
	    repr ~= ", postDiffFluxAction=[" ~ to!string(postDiffFluxAction[0]);
	    foreach (i; 1 .. postDiffFluxAction.length) {
		repr ~= ", " ~ to!string(postDiffFluxAction[i]);
	    }
	    repr ~= "]";
	}
	repr ~= ")";
	return to!string(repr);
    }

    final void applyPreReconAction(double t, int gtl, int ftl)
    {
	foreach ( gce; preReconAction ) gce.apply(t, gtl, ftl);
    }

    final void applyPostConvFluxAction(double t, int gtl, int ftl)
    {
	foreach ( bfe; postConvFluxAction ) bfe.apply(t, gtl, ftl);
    }
    
    final void applyPreSpatialDerivAction(double t, int gtl, int ftl)
    {
	foreach ( bie; preSpatialDerivAction ) bie.apply(t, gtl, ftl);
    }
    
    final void applyPostDiffFluxAction(double t, int gtl, int ftl)
    {
	foreach ( bfe; postDiffFluxAction ) bfe.apply(t, gtl, ftl);
    }

    version(implicit) {
    final void convFluxDeriv(double t, int gtl, int ftl)
    {
	final switch (blk.grid_type) {
	case Grid_t.structured_grid:
	    convFluxDeriv_structured_grid(t, gtl, ftl);
	    break;
	case Grid_t.unstructured_grid:
	    throw new Error("Implicit b.c.'s not implemented for UNSTRUCTURED GRIDS.");
	}
    }

    
    final void convFluxDeriv_structured_grid(double t, int gtl, int ftl)
    {
	size_t i, j, k;
	auto gmodel = blk.myConfig.gmodel;
	double h;
	FVInterface ifacePerturb = new FVInterface(gmodel);

	// 0th perturbation: rho
	mixin(computeBoundaryFluxDeriv("gas.rho", "0", true));
	mixin(computeBoundaryFluxDeriv("vel.refx", "1", false));
	mixin(computeBoundaryFluxDeriv("vel.refy", "2", false));
	mixin(computeBoundaryFluxDeriv("vel.refz", "3", false));
	mixin(computeBoundaryFluxDeriv("gas.e[0]", "4", true));

    }
    } // end version(implicit)
} // end class BoundaryCondition


string computeBoundaryFluxDeriv(string varName, string posInArray, bool includeThermoUpdate)
{
    string codeStr;
    codeStr ~= "final switch (which_boundary) {";
    codeStr ~= "case Face.north:";
    codeStr ~= "if ( preReconAction.length > 0 ) {";
    codeStr ~= "    applyPreReconAction(t, gtl, ftl);";
    codeStr ~= "    j = blk.jmax + 1;";
    codeStr ~= "    for (k = blk.kmin; k <= blk.kmax; ++k) {";
    codeStr ~= "        for (i = blk.imin; i <= blk.imax; ++i) {";
    codeStr ~= "            auto IFace = blk.get_ifj(i,j,k);";
    codeStr ~= "            _Lft.copy_values_from(IFace.left_cells[0].fs);";
    codeStr ~= "            h = _Lft."~varName~" * EPSILON;";
    codeStr ~= "            _Lft."~varName~" += h;";
    if ( includeThermoUpdate ) {
	codeStr ~= "           gmodel.update_thermo_from_rhoe(_Lft.gas);";
    }
    codeStr ~= "            _Rght.copy_values_from(IFace.right_cells[0].fs);";
    codeStr ~= "            ifacePerturb.copy_values_from(IFace, CopyDataOption.all);";
    codeStr ~= "            compute_interface_flux(_Lft, _Rght, ifacePerturb, gmodel, blk.omegaz);";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][0] = (ifacePerturb.F.mass - IFace.F.mass)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][1] = (ifacePerturb.F.momentum.x - IFace.F.momentum.x)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][2] = (ifacePerturb.F.momentum.y - IFace.F.momentum.y)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][3] = (ifacePerturb.F.momentum.z - IFace.F.momentum.z)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][4] = (ifacePerturb.F.total_energy - IFace.F.total_energy)/h;";
    codeStr ~= "         }";
    codeStr ~= "    }";
    codeStr ~= "}";
    codeStr ~= "else {";
    codeStr ~= "    throw new Error(\"Implicit b.c. not implemented for b.c.'s with postConvFlux actions.\");";
    codeStr ~= "}";
    codeStr ~= "break;";
    codeStr ~= "case Face.east:";
    codeStr ~= "if ( preReconAction.length > 0 ) {";
    codeStr ~= "    applyPreReconAction(t, gtl, ftl);";
    codeStr ~= "    i = blk.imax + 1;";
    codeStr ~= "    for (k = blk.kmin; k <= blk.kmax; ++k) {";
    codeStr ~= "        for (j = blk.jmin; j <= blk.jmax; ++j) {";
    codeStr ~= "            auto IFace = blk.get_ifi(i,j,k);";
    codeStr ~= "            _Lft.copy_values_from(IFace.left_cells[0].fs);";
    codeStr ~= "            h = _Lft."~varName~" * EPSILON;";
    codeStr ~= "            _Lft."~varName~" += h;";
    if ( includeThermoUpdate ) {
	codeStr ~= "           gmodel.update_thermo_from_rhoe(_Lft.gas);";
    }
    codeStr ~= "            _Rght.copy_values_from(IFace.right_cells[0].fs);";
    codeStr ~= "            ifacePerturb.copy_values_from(IFace, CopyDataOption.all);";
    codeStr ~= "            compute_interface_flux(_Lft, _Rght, ifacePerturb, gmodel, blk.omegaz);";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][0] = (ifacePerturb.F.mass - IFace.F.mass)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][1] = (ifacePerturb.F.momentum.x - IFace.F.momentum.x)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][2] = (ifacePerturb.F.momentum.y - IFace.F.momentum.y)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][3] = (ifacePerturb.F.momentum.z - IFace.F.momentum.z)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][4] = (ifacePerturb.F.total_energy - IFace.F.total_energy)/h;";
    codeStr ~= "        }";
    codeStr ~= "    }";
    codeStr ~= "}";
    codeStr ~= "else {";
    codeStr ~= "    throw new Error(\"Implicit b.c. not implemented for b.c.'s with postConvFlux actions.\");";
    codeStr ~= "}";
    codeStr ~= "break;";
    codeStr ~= "case Face.south:";
    codeStr ~= "if ( preReconAction.length > 0 ) {";
    codeStr ~= "    applyPreReconAction(t, gtl, ftl);";
    codeStr ~= "    j = blk.jmin;";
    codeStr ~= "    for (k = blk.kmin; k <= blk.kmax; ++k) {";
    codeStr ~= "        for (i = blk.imin; i <= blk.imax; ++i) {";
    codeStr ~= "            auto IFace = blk.get_ifj(i,j,k);";
    codeStr ~= "            _Lft.copy_values_from(IFace.left_cells[0].fs);";
    codeStr ~= "            _Rght.copy_values_from(IFace.right_cells[0].fs);";
    codeStr ~= "            h = _Rght."~varName~" * EPSILON;";
    codeStr ~= "            _Rght."~varName~" += h;";
    if ( includeThermoUpdate ) {
	codeStr ~= "           gmodel.update_thermo_from_rhoe(_Rght.gas);";
    }
    codeStr ~= "            ifacePerturb.copy_values_from(IFace, CopyDataOption.all);";
    codeStr ~= "            compute_interface_flux(_Lft, _Rght, ifacePerturb, gmodel, blk.omegaz);";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][0] = (ifacePerturb.F.mass - IFace.F.mass)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][1] = (ifacePerturb.F.momentum.x - IFace.F.momentum.x)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][2] = (ifacePerturb.F.momentum.y - IFace.F.momentum.y)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][3] = (ifacePerturb.F.momentum.z - IFace.F.momentum.z)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][4] = (ifacePerturb.F.total_energy - IFace.F.total_energy)/h;";
    codeStr ~= "        }";
    codeStr ~= "    }";
    codeStr ~= "}";
    codeStr ~= "else {";
    codeStr ~= "    throw new Error(\"Implicit b.c. not implemented for b.c.'s with postConvFlux actions.\");";
    codeStr ~= "}";
    codeStr ~= "break;";
    codeStr ~= "case Face.west:";
    codeStr ~= "if ( preReconAction.length > 0 ) {";
    codeStr ~= "    applyPreReconAction(t, gtl, ftl);";
    codeStr ~= "    i = blk.imin;";
    codeStr ~= "    for (k = blk.kmin; k <= blk.kmax; ++k) {";
    codeStr ~= "        for (j = blk.jmin; j <= blk.jmax; ++j) {";
    codeStr ~= "            auto IFace = blk.get_ifi(i,j,k);";
    codeStr ~= "            _Lft.copy_values_from(IFace.left_cells[0].fs);";
    codeStr ~= "            _Rght.copy_values_from(IFace.right_cells[0].fs);";
    codeStr ~= "            h = _Rght."~varName~" * EPSILON;";
    codeStr ~= "            _Rght."~varName~" += h;";
    if ( includeThermoUpdate ) {
	codeStr ~= "           gmodel.update_thermo_from_rhoe(_Rght.gas);";
    }
    codeStr ~= "            ifacePerturb.copy_values_from(IFace, CopyDataOption.all);";
    codeStr ~= "            compute_interface_flux(_Lft, _Rght, ifacePerturb, gmodel, blk.omegaz);";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][0] = (ifacePerturb.F.mass - IFace.F.mass)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][1] = (ifacePerturb.F.momentum.x - IFace.F.momentum.x)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][2] = (ifacePerturb.F.momentum.y - IFace.F.momentum.y)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][3] = (ifacePerturb.F.momentum.z - IFace.F.momentum.z)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][4] = (ifacePerturb.F.total_energy - IFace.F.total_energy)/h;";
    codeStr ~= "        }";
    codeStr ~= "    }";
    codeStr ~= "}";
    codeStr ~= "else {";
    codeStr ~= "    throw new Error(\"Implicit b.c. not implemented for b.c.'s with postConvFlux actions.\");";
    codeStr ~= "}";
    codeStr ~= "break;";
    codeStr ~= "case Face.top:";
    codeStr ~= "if ( preReconAction.length > 0 ) {";
    codeStr ~= "    applyPreReconAction(t, gtl, ftl);";
    codeStr ~= "    k = blk.kmax + 1;";
    codeStr ~= "    for (i = blk.imin; i <= blk.imax; ++i) {";
    codeStr ~= "        for (j = blk.jmin; j <= blk.jmax; ++j) {";
    codeStr ~= "            auto IFace = blk.get_ifk(i,j,k);";
    codeStr ~= "            _Lft.copy_values_from(IFace.left_cells[0].fs);";
    codeStr ~= "            h = _Lft."~varName~" * EPSILON;";
    codeStr ~= "            _Lft."~varName~" += h;";
    if ( includeThermoUpdate ) {
	codeStr ~= "           gmodel.update_thermo_from_rhoe(_Lft.gas);";
    }
    codeStr ~= "            _Rght.copy_values_from(IFace.right_cells[0].fs);";
    codeStr ~= "            ifacePerturb.copy_values_from(IFace, CopyDataOption.all);";
    codeStr ~= "            compute_interface_flux(_Lft, _Rght, ifacePerturb, gmodel, blk.omegaz);";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][0] = (ifacePerturb.F.mass - IFace.F.mass)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][1] = (ifacePerturb.F.momentum.x - IFace.F.momentum.x)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][2] = (ifacePerturb.F.momentum.y - IFace.F.momentum.y)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][3] = (ifacePerturb.F.momentum.z - IFace.F.momentum.z)/h;";
    codeStr ~= "            IFace.dFdU_L[" ~ posInArray ~ "][4] = (ifacePerturb.F.total_energy - IFace.F.total_energy)/h;";
    codeStr ~= "        }";
    codeStr ~= "    }";
    codeStr ~= "}";
    codeStr ~= "else {";
    codeStr ~= "    throw new Error(\"Implicit b.c. not implemented for b.c.'s with postConvFlux actions.\");";
    codeStr ~= "}";
    codeStr ~= "break;";
    codeStr ~= "case Face.bottom:";
    codeStr ~= "if ( preReconAction.length > 0 ) {";
    codeStr ~= "    applyPreReconAction(t, gtl, ftl);";
    codeStr ~= "    k = blk.kmin;";
    codeStr ~= "    for (i = blk.imin; i <= blk.imax; ++i) {";
    codeStr ~= "        for (j = blk.jmin; j <= blk.jmax; ++j) {";
    codeStr ~= "            auto IFace = blk.get_ifk(i,j,k);";
    codeStr ~= "            _Lft.copy_values_from(IFace.left_cells[0].fs);";
    codeStr ~= "            _Rght.copy_values_from(IFace.right_cells[0].fs);";
    codeStr ~= "            h = _Rght."~varName~" * EPSILON;";
    codeStr ~= "            _Rght."~varName~" += h;";
    if ( includeThermoUpdate ) {
	codeStr ~= "           gmodel.update_thermo_from_rhoe(_Rght.gas);";
    }
    codeStr ~= "            ifacePerturb.copy_values_from(IFace, CopyDataOption.all);";
    codeStr ~= "            compute_interface_flux(_Lft, _Rght, ifacePerturb, gmodel, blk.omegaz);";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][0] = (ifacePerturb.F.mass - IFace.F.mass)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][1] = (ifacePerturb.F.momentum.x - IFace.F.momentum.x)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][2] = (ifacePerturb.F.momentum.y - IFace.F.momentum.y)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][3] = (ifacePerturb.F.momentum.z - IFace.F.momentum.z)/h;";
    codeStr ~= "            IFace.dFdU_R[" ~ posInArray ~ "][4] = (ifacePerturb.F.total_energy - IFace.F.total_energy)/h;";
    codeStr ~= "         }";
    codeStr ~= "    }";
    codeStr ~= "}";
    codeStr ~= "else {"; 
    codeStr ~= "    throw new Error(\"Implicit b.c. not implemented for b.c.'s with postConvFlux actions.\");";
    codeStr ~= "}";
    codeStr ~= "break;";
    codeStr ~= "}";

    return codeStr;
}

