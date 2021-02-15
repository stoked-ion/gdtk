// Authors: RG, PJ, KD & IJ
// Date: 2015-11-20

module grid_motion;

import std.string;
import std.conv;
import std.algorithm;

import util.lua;
import util.lua_service;
import nm.complex;
import nm.number;
import nm.luabbla;
import lua_helper;
import fvcore;
import fvvertex;
import fvinterface;
import globalconfig;
import globaldata;
import geom;
import geom.luawrap;
import fluidblock;
import sfluidblock;
import ufluidblock;
import std.stdio;


@nogc
int set_gcl_interface_properties(SFluidBlock blk, size_t gtl, double dt) {
    FVInterface IFace;
    Vector3 pos1, pos2, temp;
    Vector3 averaged_ivel, vol;
    if (blk.myConfig.dimensions == 2) {
        FVVertex vtx1, vtx2;
        size_t k = 0;
        // loop over i-interfaces and compute interface velocity wif'.
        foreach (j; 0 .. blk.njc) {
            foreach (i; 0 .. blk.niv) {
                vtx1 = blk.get_vtx(i,j,k);
                vtx2 = blk.get_vtx(i,j+1,k);
                IFace = blk.get_ifi(i,j,k);
                pos1 = vtx1.pos[gtl];
                pos1 -= vtx2.pos[0];
                pos2 = vtx2.pos[gtl];
                pos2 -= vtx1.pos[0];
                averaged_ivel = vtx1.vel[0];
                averaged_ivel += vtx2.vel[0];
                averaged_ivel.scale(0.5);
                // Use effective edge velocity
                // Reference: D. Ambrosi, L. Gasparini and L. Vigenano
                // Full Potential and Euler solutions for transonic unsteady flow
                // Aeronautical Journal November 1994 Eqn 25
                cross(vol, pos1, pos2);
                if (blk.myConfig.axisymmetric == false) {
                    vol.scale(0.5);
                } else {
                    vol.scale(0.125*(vtx1.pos[gtl].y+vtx1.pos[0].y+vtx2.pos[gtl].y+vtx2.pos[0].y));
                }
                temp = vol; temp /= dt*IFace.area[0];
                // temp is the interface velocity (W_if) from the GCL
                // interface area determined at gtl 0 since GCL formulation
                // recommends using initial interfacial area in calculation.
                IFace.gvel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                averaged_ivel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                IFace.gvel.set(temp.z, averaged_ivel.y, averaged_ivel.z);
                averaged_ivel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                IFace.gvel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
            }
        }
        // loop over j-interfaces and compute interface velocity wif'.
        foreach (j; 0 .. blk.njv) {
            foreach (i; 0 .. blk.nic) {
                vtx1 = blk.get_vtx(i,j,k);
                vtx2 = blk.get_vtx(i+1,j,k);
                IFace = blk.get_ifj(i,j,k);
                pos1 = vtx2.pos[gtl]; pos1 -= vtx1.pos[0];
                pos2 = vtx1.pos[gtl]; pos2 -= vtx2.pos[0];
                averaged_ivel = vtx1.vel[0]; averaged_ivel += vtx2.vel[0]; averaged_ivel.scale(0.5);
                cross(vol, pos1, pos2);
                if (blk.myConfig.axisymmetric == false) {
                    vol.scale(0.5);
                } else {
                    vol.scale(0.125*(vtx1.pos[gtl].y+vtx1.pos[0].y+vtx2.pos[gtl].y+vtx2.pos[0].y));
                }
                IFace.gvel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                averaged_ivel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                if (blk.myConfig.axisymmetric && j == 0 && IFace.area[0] == 0.0) {
                    // For axi-symmetric cases the cells along the axis of symmetry have 0 interface area,
                    // this is a problem for determining Wif, so we have to catch the NaN from dividing by 0.
                    // We choose to set the y and z directions to 0, but take an averaged value for the
                    // x-direction so as to not force the grid to be stationary, defeating the moving grid's purpose.
                    IFace.gvel.set(averaged_ivel.x, to!number(0.0), to!number(0.0));
                } else {
                    temp = vol; temp /= dt*IFace.area[0];
                    IFace.gvel.set(temp.z, averaged_ivel.y, averaged_ivel.z);
                }
                averaged_ivel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                IFace.gvel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
            }
        }
    } // end if (blk.myConfig.dimensions == 2)

    // Do 3-D cases, where faces move and create a new hexahedron
    if (blk.myConfig.dimensions == 3) {
        FVVertex vtx0, vtx1, vtx2, vtx3;
        Vector3 p0, p1, p2, p3, p4, p5, p6, p7;
        Vector3 centroid_hex, sub_centroid;
        number volume, sub_volume, temp2;
        // loop over i-interfaces and compute interface velocity wif'.
        foreach (k; 0 .. blk.nkc) {
            foreach (j; 0 .. blk.njc) {
                foreach (i; 0 .. blk.niv) {
                    // Calculate volume generated by sweeping face 0123 from pos[0] to pos[gtl]
                    vtx0 = blk.get_vtx(i,j  ,k  );
                    vtx1 = blk.get_vtx(i,j+1,k  );
                    vtx2 = blk.get_vtx(i,j+1,k+1);
                    vtx3 = blk.get_vtx(i,j  ,k+1);
                    p0 = vtx0.pos[0]; p1 = vtx1.pos[0];
                    p2 = vtx2.pos[0]; p3 = vtx3.pos[0];
                    p4 = vtx0.pos[gtl]; p5 = vtx1.pos[gtl];
                    p6 = vtx2.pos[gtl]; p7 = vtx3.pos[gtl];
                    // use 6x pyramid approach as used to calculate internal volume of hex cells
                    centroid_hex.set(0.125*(p0.x+p1.x+p2.x+p3.x+p4.x+p5.x+p6.x+p7.x),
                                 0.125*(p0.y+p1.y+p2.y+p3.y+p4.y+p5.y+p6.y+p7.y),
                                 0.125*(p0.z+p1.z+p2.z+p3.z+p4.z+p5.z+p6.z+p7.z));
                    volume = 0.0;
                    pyramid_properties(p6, p7, p3, p2, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p5, p6, p2, p1, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p4, p5, p1, p0, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p7, p4, p0, p3, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p7, p6, p5, p4, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p0, p1, p2, p3, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    //
                    IFace = blk.get_ifi(i,j,k);
                    averaged_ivel = vtx0.vel[0];
                    averaged_ivel += vtx1.vel[0];
                    averaged_ivel += vtx2.vel[0];
                    averaged_ivel += vtx3.vel[0];
                    averaged_ivel.scale(0.25);
                    // Use effective face velocity, analoguous to edge velocity concept
                    // Reference: D. Ambrosi, L. Gasparini and L. Vigenano
                    // Full Potential and Euler solutions for transonic unsteady flow
                    // Aeronautical Journal November 1994 Eqn 25
                    temp2 = volume; temp /= dt*IFace.area[0];
                    // temp2 is the interface velocity (W_if) from the GCL
                    // interface area determined at gtl 0 since GCL formulation
                    // recommends using initial interfacial area in calculation.
                    IFace.gvel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                    averaged_ivel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                    IFace.gvel.set(temp2, averaged_ivel.y, averaged_ivel.z);
                    averaged_ivel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                    IFace.gvel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                }
            }
        }
        // loop over j-interfaces and compute interface velocity wif'.
        foreach (k; 0 .. blk.nkc) {
            foreach (j; 0 .. blk.njv) {
                foreach (i; 0 .. blk.nic) {
                    // Calculate volume generated by sweeping face 0123 from pos[0] to pos[gtl]
                    vtx0 = blk.get_vtx(i  ,j,k  );
                    vtx1 = blk.get_vtx(i  ,j,k+1);
                    vtx2 = blk.get_vtx(i+1,j,k+1);
                    vtx3 = blk.get_vtx(i+1,j,k  );
                    p0 = vtx0.pos[0]; p1 = vtx1.pos[0];
                    p2 = vtx2.pos[0]; p3 = vtx3.pos[0];
                    p4 = vtx0.pos[gtl]; p5 = vtx1.pos[gtl];
                    p6 = vtx2.pos[gtl]; p7 = vtx3.pos[gtl];
                    // use 6x pyramid approach as used to calculate internal volume of hex cells
                    centroid_hex.set(0.125*(p0.x+p1.x+p2.x+p3.x+p4.x+p5.x+p6.x+p7.x),
                                 0.125*(p0.y+p1.y+p2.y+p3.y+p4.y+p5.y+p6.y+p7.y),
                                 0.125*(p0.z+p1.z+p2.z+p3.z+p4.z+p5.z+p6.z+p7.z));
                    volume = 0.0;
                    pyramid_properties(p6, p7, p3, p2, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p5, p6, p2, p1, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p4, p5, p1, p0, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p7, p4, p0, p3, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p7, p6, p5, p4, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p0, p1, p2, p3, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    //
                    IFace = blk.get_ifj(i,j,k);
                    averaged_ivel = vtx0.vel[0];
                    averaged_ivel += vtx1.vel[0];
                    averaged_ivel += vtx2.vel[0];
                    averaged_ivel += vtx3.vel[0];
                    averaged_ivel.scale(0.25);
                    // Use effective face velocity, analoguous to edge velocity concept
                    // Reference: D. Ambrosi, L. Gasparini and L. Vigenano
                    // Full Potential and Euler solutions for transonic unsteady flow
                    // Aeronautical Journal November 1994 Eqn 25
                    temp2 = volume; temp /= dt*IFace.area[0];
                    // temp2 is the interface velocity (W_if) from the GCL
                    // interface area determined at gtl 0 since GCL formulation
                    // recommends using initial interfacial area in calculation.
                    IFace.gvel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                    averaged_ivel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                    IFace.gvel.set(temp2, averaged_ivel.y, averaged_ivel.z);
                    averaged_ivel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                    IFace.gvel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                }
            }
        }
        // loop over k-interfaces and compute interface velocity wif'.
        foreach (k; 0 .. blk.nkv) {
            foreach (j; 0 .. blk.njc) {
                foreach (i; 0 .. blk.nic) {
                    // Calculate volume generated by sweeping face 0123 from pos[0] to pos[gtl]
                    vtx0 = blk.get_vtx(i  ,j  ,k);
                    vtx1 = blk.get_vtx(i+1,j  ,k);
                    vtx2 = blk.get_vtx(i+1,j+1,k);
                    vtx3 = blk.get_vtx(i  ,j+1,k);
                    p0 = vtx0.pos[0]; p1 = vtx1.pos[0];
                    p2 = vtx2.pos[0]; p3 = vtx3.pos[0];
                    p4 = vtx0.pos[gtl]; p5 = vtx1.pos[gtl];
                    p6 = vtx2.pos[gtl]; p7 = vtx3.pos[gtl];
                    // use 6x pyramid approach as used to calculate internal volume of hex cells
                    centroid_hex.set(0.125*(p0.x+p1.x+p2.x+p3.x+p4.x+p5.x+p6.x+p7.x),
                                 0.125*(p0.y+p1.y+p2.y+p3.y+p4.y+p5.y+p6.y+p7.y),
                                 0.125*(p0.z+p1.z+p2.z+p3.z+p4.z+p5.z+p6.z+p7.z));
                    volume = 0.0;
                    pyramid_properties(p6, p7, p3, p2, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p5, p6, p2, p1, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p4, p5, p1, p0, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p7, p4, p0, p3, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p7, p6, p5, p4, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    pyramid_properties(p0, p1, p2, p3, centroid_hex, sub_centroid, sub_volume);
                    volume += sub_volume;
                    //
                    IFace = blk.get_ifk(i,j,k);
                    averaged_ivel = vtx0.vel[0];
                    averaged_ivel += vtx1.vel[0];
                    averaged_ivel += vtx2.vel[0];
                    averaged_ivel += vtx3.vel[0];
                    averaged_ivel.scale(0.25);
                    // Use effective face velocity, analoguous to edge velocity concept
                    // Reference: D. Ambrosi, L. Gasparini and L. Vigenano
                    // Full Potential and Euler solutions for transonic unsteady flow
                    // Aeronautical Journal November 1994 Eqn 25
                    temp2 = volume; temp2 /= dt*IFace.area[0];
                    // temp2 is the interface velocity (W_if) from the GCL
                    // interface area determined at gtl 0 since GCL formulation
                    // recommends using initial interfacial area in calculation.
                    IFace.gvel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                    averaged_ivel.transform_to_local_frame(IFace.n, IFace.t1, IFace.t2);
                    IFace.gvel.set(temp2, averaged_ivel.y, averaged_ivel.z);
                    averaged_ivel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                    IFace.gvel.transform_to_global_frame(IFace.n, IFace.t1, IFace.t2);
                }
            }
        }
    }
    return 0;
} // end set_gcl_interface_properties()

@nogc
void predict_vertex_positions(SFluidBlock blk, double dt, int gtl)
{
    foreach (vtx; blk.vertices) {
        if (gtl == 0) {
            // predictor/sole step; update grid
            vtx.pos[1] = vtx.pos[0] + dt * vtx.vel[0];
        } else {
            // corrector step; keep grid fixed
            vtx.pos[2] = vtx.pos[1];
        }
    }
    return;
} // end predict_vertex_positions()
