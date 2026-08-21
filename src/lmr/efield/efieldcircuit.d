/**
 * External-circuit network for the low-Rm efield solver (Path 1, Phase 1).
 *
 * Motivation
 * ----------
 * Every electrode boundary condition in the solver today (SheathField) holds its
 * metal potential at a value fixed by the Lua input file. That is a faithful model
 * of an electrode wired to its own ideal, zero-impedance supply, which is what a
 * Faraday channel (independent pairs) and a diagonal channel (resistor ladder with
 * negligible tap impedance) physically are.
 *
 * It is NOT a model of electrodes wired TO EACH OTHER. Shorting two electrodes does
 * not mean "hold both at the same prescribed potential" -- it means "let current
 * flow between them subject to Kirchhoff's current law, through whatever impedance
 * the actual conductor has". No prescribed-potential BC can express that, because
 * there is no equation anywhere in the assembled system coupling the current at one
 * electrode to the current at another.
 *
 * This module supplies the missing equation. Each independent electrode terminal
 * becomes a circuit NODE with its own unknown potential q_m, and the network of
 * resistors between nodes (and from nodes to supplies) contributes Kirchhoff rows
 * that are solved simultaneously with the field.
 *
 * Scope (deliberate)
 * ------------------
 * Resistors and ideal voltage sources only. The physical devices being modelled are
 * ballast resistors, bus bars and pulsed supplies; a resistor ladder is sufficient
 * for every case this project needs. This is NOT a general circuit simulator and
 * should not grow into one.
 *
 * Phase 1 scope: data structures + the nodal-analysis stamp (assemble_L_and_c).
 * Nothing here is wired into the field assembly yet -- see efield.d, Phase 2.
 *
 * Author: (gdtk-mhd) 2026
 */
module lmr.efield.efieldcircuit;

import std.conv;
import std.format;
import std.math;
import std.algorithm : canFind;

/*
    One electrode terminal. Faces are associated with a node by the CircuitElectrode
    boundary condition (efieldbc.d) naming this node's id; the association is resolved
    during matrix assembly, so no face references are stored here. Two electrode
    groups wired together (e.g. the north and south segment of a Hall pair) simply
    name the SAME node id -- that shared id IS the short.
*/
struct CircuitNode {
    int id;
    double nominal_voltage; // initial estimate of q_m, and the fallback before the
                            // first solve (the sheath linearization needs a finite
                            // value to linearize about; see SheathField's NaN guard).
    string label;           // for diagnostics only
}

/*
    A resistor between two nodes, or between a node and a fixed supply.
    b < 0 marks a leg to a supply held at V_supply (a "ground" leg is just
    V_supply = 0). R is in ohms for a per-metre-depth 2D problem, i.e. ohm.m --
    consistent with currents being A/m in this solver.
*/
struct CircuitResistor {
    int a;
    int b;             // < 0 => leg to a fixed supply
    double R;
    double V_supply;   // only meaningful when b < 0
}

class ExternalCircuit {
    this() {}

    // ---- construction -----------------------------------------------------
    int addNode(double nominal_voltage, string label="") {
        int id = to!int(nodes.length);
        nodes ~= CircuitNode(id, nominal_voltage, label);
        _q ~= nominal_voltage;
        return id;
    }

    void addResistor(int a, int b, double R) {
        checkNode(a); checkNode(b);
        checkR(R);
        resistors ~= CircuitResistor(a, b, R, 0.0);
    }

    void addSupplyLeg(int a, double R, double V_supply) {
        checkNode(a);
        checkR(R);
        resistors ~= CircuitResistor(a, -1, R, V_supply);
    }

    // ---- the nodal-analysis stamp ------------------------------------------
    /*
        Build the K x K conductance matrix L and the K-vector c of independent
        sources, for the circuit block of the augmented system

            [ A0   Uc ] [ phi ]   [ b0 ]
            [ Wt   L  ] [  q  ] = [ c  ]

        Row m is Kirchhoff's current law at node m:

            sum_{faces in m} S*J(phi_cell - q_m)  +  sum_{legs at m} (q_m - q_other)/R  =  0

        The plasma term (first sum) is contributed by the field assembly into Wt and
        into L's diagonal; this routine supplies ONLY the external network terms
        (second sum). Standard stamping:

          node-to-node resistor R between a and b:
              L[a,a] += 1/R ;  L[b,b] += 1/R ;  L[a,b] -= 1/R ;  L[b,a] -= 1/R
          supply leg R from node a to a source held at V:
              L[a,a] += 1/R ;  c[a]   += V/R

        L is returned row-major, length K*K. Both L and c are ZEROED here, so the
        caller must add the plasma contributions AFTER calling this.
    */
    void assemble_L_and_c(ref double[] L, ref double[] c) const {
        immutable size_t K = nodes.length;
        L.length = K*K; L[] = 0.0;
        c.length = K;   c[] = 0.0;
        foreach (r; resistors) {
            double g = 1.0/r.R;
            if (r.b < 0) {
                // leg to a fixed supply
                L[r.a*K + r.a] += g;
                c[r.a]         += g*r.V_supply;
            } else {
                L[r.a*K + r.a] += g;
                L[r.b*K + r.b] += g;
                L[r.a*K + r.b] -= g;
                L[r.b*K + r.a] -= g;
            }
        }
    }

    /*
        A node with no galvanic path to any fixed supply leaves the circuit block
        singular (its potential is only defined up to a constant, exactly the classic
        floating-ground problem). The field's own sheath terms usually anchor it in
        practice, but a network that is floating BY CONSTRUCTION is almost always an
        input error, so flag it early with a clear message rather than letting GMRES
        or the dense solve fail obscurely later.

        Returns true if every node can reach a supply leg through the resistor graph.
    */
    bool isGrounded(ref string report) const {
        immutable size_t K = nodes.length;
        if (K == 0) { report = "circuit has no nodes"; return false; }
        auto seen = new bool[K];
        int[] stack;
        foreach (r; resistors) if (r.b < 0 && !seen[r.a]) { seen[r.a] = true; stack ~= r.a; }
        while (stack.length > 0) {
            int n = stack[$-1]; stack = stack[0 .. $-1];
            foreach (r; resistors) {
                if (r.b < 0) continue;
                int other = -1;
                if (r.a == n) other = r.b;
                else if (r.b == n) other = r.a;
                if (other >= 0 && !seen[other]) { seen[other] = true; stack ~= other; }
            }
        }
        foreach (i, s; seen) {
            if (!s) {
                report = format("circuit node %d (%s) has no resistive path to any supply leg; "
                                ~ "the network is floating and its potential is undetermined",
                                i, nodes[i].label);
                return false;
            }
        }
        report = "";
        return true;
    }

    // ---- state -------------------------------------------------------------
    @property size_t nnodes() const { return nodes.length; }
    @property const(double)[] q() const { return _q; }
    void setQ(const double[] qnew) {
        if (qnew.length != _q.length)
            throw new Error(format("ExternalCircuit.setQ length %d != nnodes %d",
                                   qnew.length, _q.length));
        _q[] = qnew[];
    }
    double q_of(int node_id) const {
        checkNode(node_id);
        return _q[node_id];
    }

    override string toString() const {
        string s = format("ExternalCircuit(%d nodes, %d legs)\n", nodes.length, resistors.length);
        foreach (n; nodes) s ~= format("  node %d '%s' nominal=%g q=%g\n",
                                       n.id, n.label, n.nominal_voltage, _q[n.id]);
        foreach (r; resistors) {
            if (r.b < 0) s ~= format("  leg  %d -> supply %g V through %g ohm\n", r.a, r.V_supply, r.R);
            else         s ~= format("  leg  %d -- %d through %g ohm\n", r.a, r.b, r.R);
        }
        return s;
    }

private:
    void checkNode(int i) const {
        if (i < 0 || i >= to!int(nodes.length))
            throw new Error(format("circuit node id %d out of range [0,%d)", i, nodes.length));
    }
    static void checkR(double R) {
        if (!(R > 0.0) || R != R)
            throw new Error(format("circuit resistance must be finite and > 0, got %g", R));
    }

    CircuitNode[] nodes;
    CircuitResistor[] resistors;
    double[] _q;
}

version(unittest) {
    import std.stdio;
    import std.math : abs, isClose;

    // Helper: fetch L[i,j] from the row-major buffer.
    private double Lij(const double[] L, size_t K, size_t i, size_t j) { return L[i*K + j]; }
}

// Example A of the plan (Sec. 2.6): independent segments, each with its own supply
// through its own ballast resistor. L must be diagonal (no node-to-node coupling)
// and c must be V/R per node.
unittest {
    auto ck = new ExternalCircuit();
    int n0 = ck.addNode(400.0, "seg0");
    int n1 = ck.addNode(400.0, "seg1");
    ck.addSupplyLeg(n0, 2.0, 400.0);   // 1/R = 0.5, V/R = 200
    ck.addSupplyLeg(n1, 4.0, 100.0);   // 1/R = 0.25, V/R = 25
    double[] L, c;
    ck.assemble_L_and_c(L, c);
    immutable size_t K = 2;
    assert(isClose(Lij(L,K,0,0), 0.5),  "A: L[0,0]");
    assert(isClose(Lij(L,K,1,1), 0.25), "A: L[1,1]");
    assert(Lij(L,K,0,1) == 0.0 && Lij(L,K,1,0) == 0.0, "A: L must be diagonal (segments independent)");
    assert(isClose(c[0], 200.0), "A: c[0]");
    assert(isClose(c[1], 25.0),  "A: c[1]");
    string rep;
    assert(ck.isGrounded(rep), "A: every node has a supply leg, should be grounded");
}

// Example B of the plan (Sec. 2.6): a 3-node Hall bus. Adjacent nodes tied by a
// bus-bar resistance; the two END nodes additionally tied to the axial supply.
// This is the topology that supplies the current-limiting equation missing from
// every previous Hall attempt.
unittest {
    auto ck = new ExternalCircuit();
    int a = ck.addNode(0.0, "ax0");
    int b = ck.addNode(0.0, "ax1");
    int d = ck.addNode(0.0, "ax2");
    ck.addResistor(a, b, 0.5);          // g = 2
    ck.addResistor(b, d, 0.25);         // g = 4
    ck.addSupplyLeg(a, 1.0,   0.0);     // g = 1,  V/R = 0
    ck.addSupplyLeg(d, 0.2, -150.0);    // g = 5,  V/R = -750
    double[] L, c;
    ck.assemble_L_and_c(L, c);
    immutable size_t K = 3;
    // interior node b sees both bus legs and no supply leg
    assert(isClose(Lij(L,K,1,1), 6.0), "B: L[1,1] = 2 + 4");
    assert(isClose(Lij(L,K,0,0), 3.0), "B: L[0,0] = 2 (bus) + 1 (supply)");
    assert(isClose(Lij(L,K,2,2), 9.0), "B: L[2,2] = 4 (bus) + 5 (supply)");
    // off-diagonals are the negated bus conductances, and L must be symmetric
    assert(isClose(Lij(L,K,0,1), -2.0) && isClose(Lij(L,K,1,0), -2.0), "B: L[0,1]");
    assert(isClose(Lij(L,K,1,2), -4.0) && isClose(Lij(L,K,2,1), -4.0), "B: L[1,2]");
    assert(Lij(L,K,0,2) == 0.0 && Lij(L,K,2,0) == 0.0, "B: non-adjacent nodes must not couple");
    // only the supply legs contribute to c
    assert(isClose(c[0], 0.0),     "B: c[0]");
    assert(c[1] == 0.0,            "B: c[1] (interior node has no source)");
    assert(isClose(c[2], -750.0),  "B: c[2]");
    string rep;
    assert(ck.isGrounded(rep), "B: reachable through the bus, should be grounded");
}

// A passive resistor network conserves charge: each node-to-node stamp must have
// zero row sum, so L*1 picks up ONLY the supply-leg conductances. This is a
// property-based check that catches sign/transposition slips the two hand-computed
// cases above could miss.
unittest {
    auto ck = new ExternalCircuit();
    foreach (i; 0 .. 4) ck.addNode(0.0, format("n%d", i));
    ck.addResistor(0, 1, 3.0);
    ck.addResistor(1, 2, 7.0);
    ck.addResistor(2, 3, 11.0);
    ck.addResistor(0, 3, 13.0);
    ck.addSupplyLeg(2, 5.0, 12.0);
    double[] L, c;
    ck.assemble_L_and_c(L, c);
    immutable size_t K = 4;
    foreach (i; 0 .. K) {
        double rowsum = 0.0;
        foreach (j; 0 .. K) rowsum += Lij(L, K, i, j);
        double expect = (i == 2) ? 1.0/5.0 : 0.0;  // only node 2 has a supply leg
        assert(abs(rowsum - expect) < 1.0e-12,
               format("row sum of L must equal the node's supply conductance (node %d)", i));
    }
    // symmetry
    foreach (i; 0 .. K) foreach (j; 0 .. K)
        assert(abs(Lij(L,K,i,j) - Lij(L,K,j,i)) < 1.0e-14, "L must be symmetric");
}

// A floating sub-network must be reported, not silently accepted.
unittest {
    auto ck = new ExternalCircuit();
    int a = ck.addNode(0.0, "tied");
    int b = ck.addNode(0.0, "floating");
    ck.addSupplyLeg(a, 1.0, 10.0);
    // node b deliberately left unconnected
    string rep;
    assert(!ck.isGrounded(rep), "a node with no path to a supply must be flagged");
    assert(rep.length > 0 && rep.canFind("floating"), "report should name the offending node");
    // and once bridged, it should pass
    ck.addResistor(a, b, 2.0);
    assert(ck.isGrounded(rep), "bridging to a grounded node should satisfy the check");
}

// Guard rails: bad input should fail loudly at construction, not produce a silently
// wrong matrix (a zero or negative resistance is an infinite/negative conductance).
unittest {
    auto ck = new ExternalCircuit();
    int a = ck.addNode(0.0);
    bool threw = false;
    try { ck.addSupplyLeg(a, 0.0, 10.0); } catch (Error e) { threw = true; }
    assert(threw, "zero resistance must throw");
    threw = false;
    try { ck.addResistor(a, 99, 1.0); } catch (Error e) { threw = true; }
    assert(threw, "out-of-range node id must throw");
    threw = false;
    try { ck.setQ([1.0, 2.0]); } catch (Error e) { threw = true; }
    assert(threw, "setQ with wrong length must throw");
}
