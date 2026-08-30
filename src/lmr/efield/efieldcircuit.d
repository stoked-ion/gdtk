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

/**
 * A current source injecting I_supply (A per metre of depth) into node `a`.
 *
 * Real Hall accelerators are driven current-controlled, not voltage-controlled, and the
 * distinction is not cosmetic here: with a voltage source the terminal sheaths absorb almost
 * all of the applied drive (measured: 94% of it, leaving the plasma only 6%), so a
 * voltage-driven Hall connection is sheath-limited rather than field-limited. Specifying the
 * current instead moves the sheath drop into the *answer* rather than the input.
 *
 * A current source contributes ONLY to the right-hand side -- it adds no conductance -- so it
 * cannot anchor the potential level. A current-driven circuit therefore still needs one
 * voltage reference; use a high-resistance supply leg, which fixes the gauge while carrying
 * negligible current.
 */
struct CircuitCurrentSource {
    int a;
    double I_supply;
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

    void addCurrentSource(int a, double I_supply) {
        checkNode(a);
        sources ~= CircuitCurrentSource(a, I_supply);
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
          current source injecting I into node a:
              c[a] += I                       (no conductance -- see CircuitCurrentSource)

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
        foreach (src; sources) c[src.a] += src.I_supply;
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
    /*
        Solvability of the augmented system.

        A node's potential q_m is determined if its Schur-complement row has a
        non-trivial diagonal. Two independent mechanisms supply one:

          (i)  the RESISTOR NETWORK -- a path of resistors from the node to a leg
               that terminates on a fixed supply, contributing +1/R to L[m,m];
          (ii) the PLASMA -- any electrode face on the node stamps
               L[m,m] += -S*Jp through sheathFaceStamp, i.e. the sheath's
               differential conductance ties the metal potential to the adjacent
               cell potential.

        Mechanism (ii) is what makes a FLOATING electrode pair well-posed, and it
        is essential for the physical Hall connection: intermediate segment pairs
        are deliberately isolated from each other and from the supply (that is what
        segmentation MEANS -- see the note in Sec. 2.6 of the Path 1 plan), and each
        such pair simply floats to the potential at which its net sheath current is
        zero. Requiring a resistive path to a supply for EVERY node -- as this check
        originally did -- would have rejected exactly the topology the Hall case
        needs.

        Current sources do NOT count for this: they stamp only the right-hand side, so
        adding a constant to every potential still leaves the system unchanged. A
        current-driven circuit needs a voltage reference, and the natural one is a
        high-resistance supply leg -- it sets the level while drawing almost no current.

        What is still genuinely required is at least ONE supply leg somewhere in the
        circuit. Without it, nothing anchors the absolute level: the sheath stamps
        are all differences (phi_cell - q_m), so adding a constant to every phi and
        every q leaves the whole augmented system unchanged, and it is singular.

        `electrodeFaceCount[m]` must be the GLOBAL count over all MPI ranks -- a
        node's faces can live entirely on another rank.
    */
    bool isGrounded(ref string report, const(size_t)[] electrodeFaceCount = null) const {
        immutable size_t K = nodes.length;
        if (K == 0) { report = "circuit has no nodes"; return false; }
        bool anySupply = false;
        foreach (r; resistors) if (r.b < 0) { anySupply = true; break; }
        if (!anySupply) {
            report = "circuit has no supply leg; every node potential is defined only "
                   ~ "up to a common additive constant and the augmented system is singular";
            return false;
        }
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
            if (s) continue;
            if (electrodeFaceCount !is null && i < electrodeFaceCount.length
                && electrodeFaceCount[i] > 0) continue; // tied to the plasma by its sheath
            report = format("circuit node %d (%s) has neither a resistive path to a supply leg "
                            ~ "nor any electrode face; its potential is undetermined",
                            i, nodes[i].label);
            return false;
        }
        report = "";
        return true;
    }

    // ---- state -------------------------------------------------------------
    @property size_t nnodes() const { return nodes.length; }

    /**
     * Total electrical power delivered by the supplies, and the current in each supply
     * leg, given the solved node potentials.
     *
     * The leg current is (V_supply - q_a)/R, positive into the node, so the power the
     * supply delivers is V_supply times that. This is the only place the SUPPLIED power
     * can be got exactly: it is topology dependent, and V*|I| with a single nominal V is
     * only a proxy that happens to be right for a two-terminal connection. Comparing
     * electrode configurations -- Faraday against Hall against diagonal -- needs the real
     * number, because those topologies differ in precisely this.
     */
    /// Total current injected by the current sources (A per metre of depth).
    double sourceCurrent() const {
        double I = 0.0;
        foreach (src; sources) if (src.I_supply > 0.0) I += src.I_supply;
        return I;
    }

    double supplyPower(ref double[] legCurrents) const {
        legCurrents.length = 0;
        double P = 0.0;
        foreach (r; resistors) {
            if (r.b >= 0) continue;                 // node-to-node, not a supply leg
            double I = (r.V_supply - _q[r.a])/r.R;  // A per metre of depth, into the node
            legCurrents ~= I;
            P += r.V_supply*I;
        }
        return P;
    }
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
    CircuitCurrentSource[] sources;
    double[] _q;
}

/*
    Build an ExternalCircuit from the config JSON emitted by output.lua:
        {"nodes": [{"nominal_voltage": .., "label": ".."}, ..],
         "resistors": [{"a": i, "b": j, "R": .., "V_supply": ..}, ..],
         "sources":   [{"a": i, "I_supply": ..}, ..]}
    b < 0 marks a leg to a fixed supply. Returns null when there are no nodes, which
    is the signal to the field solver to take its ordinary (non-circuit) path.
*/
ExternalCircuit create_external_circuit(string json_text)
{
    import std.json : parseJSON, JSONValue, JSONType;
    if (json_text.length == 0) return null;
    JSONValue j;
    try { j = parseJSON(json_text); }
    catch (Exception e) {
        throw new Error("could not parse config external_circuit as JSON: " ~ e.msg);
    }
    if (("nodes" !in j) || j["nodes"].type != JSONType.array) return null;
    if (j["nodes"].array.length == 0) return null;

    auto ck = new ExternalCircuit();
    foreach (n; j["nodes"].array) {
        double v = ("nominal_voltage" in n) ? n["nominal_voltage"].get!double : 0.0;
        string lab = ("label" in n) ? n["label"].str : "";
        ck.addNode(v, lab);
    }
    if ("sources" in j && j["sources"].type == JSONType.array) {
        foreach (src; j["sources"].array) {
            ck.addCurrentSource(src["a"].get!int,
                                ("I_supply" in src) ? src["I_supply"].get!double : 0.0);
        }
    }
    if ("resistors" in j && j["resistors"].type == JSONType.array) {
        foreach (r; j["resistors"].array) {
            int a = r["a"].get!int;
            int b = ("b" in r) ? r["b"].get!int : -1;
            double R = r["R"].get!double;
            if (b < 0) {
                double V = ("V_supply" in r) ? r["V_supply"].get!double : 0.0;
                ck.addSupplyLeg(a, R, V);
            } else {
                ck.addResistor(a, b, R);
            }
        }
    }
    return ck;
}

/*
    THE SIGN CONVENTION, IN ONE PLACE.

    Every contribution the external circuit makes to the augmented system comes
    through this one function. efield.d calls it during assembly; the unit tests
    below call the SAME function and check it against an independent dense solve.
    If the convention is wrong, the tests fail -- it cannot drift out of agreement
    with the code that uses it, because there is only one copy.

    Augmented system (NO minus folded into Uc, so there is no sign to misremember):

        [ A0  Uc ] [ phi ]   [ b0 ]
        [ Wt  L  ] [  q  ] = [ c  ]

    For one sheath face of area S on cell k belonging to circuit node m, with the
    sheath law linearized about the frozen prior iterate (phi_star, q_star):

        J(phi_k, q_m) ~= J0 + Jp*(phi_k - phi_star) - Jp*(q_m - q_star)

    The face contributes -S*J to cell k's charge balance, and +S*J to node m's
    Kirchhoff sum (the same current leaves the plasma and enters the electrode, so
    the two rows see equal and opposite contributions).

    Matching coefficients gives the six stamps below. The `const` term is the piece
    that is easy to drop: without its +S*Jp*q_star part the cell row does NOT reduce
    to SheathField.linearized_robin when q_m is frozen, which unittest
    `circuit stamp reduces to SheathField when the node is frozen` checks directly.
*/
void sheathFaceStamp(double S, double J0, double Jp, double phi_star, double q_star,
                     out double a_diag,  // -> A0[k,k]
                     out double u_coeff, // -> Uc[k,m]
                     out double b_cell,  // -> b0[k]
                     out double wt_coeff,// -> Wt[m,k]
                     out double l_diag,  // -> L[m,m]
                     out double c_node)  // -> c[m]
{
    immutable double konst = S*(J0 - Jp*phi_star + Jp*q_star);
    // Cell row. efield.d assembles A*phi = b, so the residual contribution is
    // a_diag*phi + u_coeff*q - b_cell, and it must equal -S*J_lin.
    a_diag   = -S*Jp;
    u_coeff  =  S*Jp;
    b_cell   =  konst;
    // Node row. Kirchhoff at the electrode: the plasma current INTO the electrode is
    // +S*J (equal and opposite to the cell's -S*J), and the network contributes
    // (q - V)/R. assemble_L_and_c stamps the network as L += +1/R, c += +V/R, so the
    // sheath part must carry the sign that is consistent with THAT, not merely the
    // negation of the cell row.
    wt_coeff =  S*Jp;
    l_diag   = -S*Jp;
    c_node   = -konst;
}

/*
    The Schur/Woodbury solve of the augmented system.

    A0 is never modified and never combined with L -- that is the whole point. The
    caller supplies `solveA0`, a delegate that solves A0*x = rhs (GMRES + ILU(0) in
    efield.d; a dense factorization in the unit tests). This keeps the algebra
    testable without the fluid solver, and means the tested code path is literally
    the one that runs in production.

        X  = A0^-1 Uc                (K solves)
        y0 = A0^-1 b0                (1 solve -- exactly today's field solve)
        S  = L - Wt X                (K x K, dense, tiny)
        q  = S^-1 (c - Wt y0)
        phi= y0 - X q

    Cost: K+1 solves of the unmodified system, plus one K x K dense solve.

    Note on conditioning (measured, see tools/woodbury/README.md): cond(A0) and
    cond(S) stay O(1e2) even as supply resistances go to zero, while the COMBINED
    (N+K)x(N+K) matrix reaches 1e13. Because this routine never forms that combined
    matrix, it is the better-conditioned formulation, not merely the cheaper one.
*/
void schurSolve(double[] delegate(const(double)[]) solveA0,
                const double[][] Uc,      // [K][N_local] -- column m of Uc as Uc[m]
                const double[][] Wt,      // [K][N_local]
                const double[] L,         // K*K row-major, ALREADY reduced across ranks
                const double[] b0,        // N_local
                const double[] c,         // K, ALREADY reduced across ranks
                ref double[] phi,         // out, N_local
                ref double[] q,           // out, K
                void delegate(double[]) allreduceSum = null)
{
    /*
        MPI note. N is the LOCAL cell count: every rank owns its own slice of the
        domain, so Uc[m] and Wt[m] index THIS rank's cells and must never be summed
        element-wise across ranks (doing so adds together unrelated cells and
        silently corrupts the coupling).

        What IS global is the dot products Wt.X and Wt.y0, which are sums over every
        cell in the domain. Those partial sums are formed locally here and combined
        through `allreduceSum`, along with the K x K Schur complement. L and c must
        already be reduced by the caller, since their sheath contributions come from
        faces that may live on any rank.
    */
    import nm.bbla : Matrix, gaussJordanElimination;
    immutable size_t K = Uc.length;
    immutable size_t N = b0.length;

    // X's columns, one solve of the untouched system each
    double[][] X; X.length = K;
    foreach (m; 0 .. K) X[m] = solveA0(Uc[m]).dup;
    double[] y0 = solveA0(b0).dup;

    // Schur complement and its RHS. Caller must have already reduced Uc/Wt/L/c
    // across MPI ranks if the problem is distributed.
    // local partial sums of the global dot products Wt.X and Wt.y0
    auto packed = new double[K*K + K];
    foreach (i; 0 .. K) {
        foreach (j; 0 .. K) {
            double wtx = 0.0;
            foreach (n; 0 .. N) wtx += Wt[i][n]*X[j][n];
            packed[i*K + j] = wtx;
        }
        double wty = 0.0;
        foreach (n; 0 .. N) wty += Wt[i][n]*y0[n];
        packed[K*K + i] = wty;
    }
    if (allreduceSum !is null) allreduceSum(packed);

    auto Aug = new Matrix!double(K, K+1);
    foreach (i; 0 .. K) {
        foreach (j; 0 .. K) Aug[i, j] = L[i*K + j] - packed[i*K + j];
        Aug[i, K] = c[i] - packed[K*K + i];
    }
    gaussJordanElimination!double(Aug);

    q.length = K;
    foreach (i; 0 .. K) q[i] = Aug[i, K];
    phi.length = N;
    foreach (n; 0 .. N) {
        double xq = 0.0;
        foreach (m; 0 .. K) xq += X[m][n]*q[m];
        phi[n] = y0[n] - xq;
    }
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

/*
    The Hall topology: an electrode pair that is isolated from BOTH the supply and
    every other node is still well-posed, because its own sheath faces stamp
    L[m,m] += -S*Jp. This is the case the original network-only check wrongly
    rejected, and it is exactly the intermediate segment pair of a segmented Hall
    channel. Pinned here so a future tightening of isGrounded cannot silently break
    the Hall case.
*/
unittest {
    auto ck = new ExternalCircuit();
    int a = ck.addNode(400.0, "end_driven");
    int f = ck.addNode(200.0, "mid_floating");
    ck.addSupplyLeg(a, 1.0e-3, 400.0);
    string rep;
    // With no face information, the floating node cannot be justified.
    assert(!ck.isGrounded(rep), "floating node with no faces must still be rejected");
    // Given electrode faces on it, it IS determined -- by the plasma, not the network.
    size_t[] nf = [10, 10];
    assert(ck.isGrounded(rep, nf),
           "a floating node carrying electrode faces is tied to the plasma by its sheath");
    // ... but a node with neither a path nor faces is not.
    size_t[] nf_none = [10, 0];
    assert(!ck.isGrounded(rep, nf_none), "no path and no faces => undetermined");
    assert(rep.canFind("mid_floating"), "report should name the offending node");
}

/*
    At least one supply leg is required no matter how many faces exist: the sheath
    stamps depend only on (phi_cell - q_m), so a circuit with no supply leaves the
    whole augmented system invariant under a uniform shift of phi and q. Adding
    faces must NOT be able to paper over that -- checked explicitly, because the
    face-count relaxation above is precisely the kind of change that could.
*/
unittest {
    auto ck = new ExternalCircuit();
    int a = ck.addNode(0.0, "n0");
    int b = ck.addNode(0.0, "n1");
    ck.addResistor(a, b, 1.0e-3);   // nodes tied to each other, but to no supply
    string rep;
    size_t[] nf = [100, 100];
    assert(!ck.isGrounded(rep, nf),
           "a supply-free circuit is gauge-singular however many faces it carries");
    assert(rep.canFind("supply"), "report should identify the missing supply leg");
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

// ---------------------------------------------------------------------------
// Convention verification.
//
// These tests exist specifically to make the sign convention FALSIFIABLE rather
// than merely documented. They call sheathFaceStamp and schurSolve -- the same
// functions efield.d calls -- and check them against an independent dense solve
// of the full augmented system. A convention error cannot pass them.
// ---------------------------------------------------------------------------

// Identity 1: with the node frozen, the CELL row must reduce to exactly what
// efield.d already assembles for SheathField. Note this is checked against
// SheathField's real convention -- A[k,k] += a_diag and b[k] += b_rhs in an
// A*phi = b system -- not against an internally-chosen sign, because an earlier
// version of these tests was self-consistent and still disagreed with efield.d.
unittest {
    immutable double S = 1.7, J0 = -0.35, Jp = 0.82, phi_star = 3.1, q_star = 411.0;
    double a_diag, u_coeff, b_cell, wt_coeff, l_diag, c_node;
    sheathFaceStamp(S, J0, Jp, phi_star, q_star,
                    a_diag, u_coeff, b_cell, wt_coeff, l_diag, c_node);

    // What SheathField.linearized_robin returns for the same face, with
    // Velectrode = q_star (efield.d does: A[k,k] += a_diag; b[k] += b_rhs).
    immutable double sf_a_diag = -S*Jp;
    immutable double sf_b_rhs  =  S*(J0 - Jp*phi_star);

    // Freezing q at q_star moves the Uc term to the right-hand side.
    assert(abs(a_diag - sf_a_diag) < 1.0e-12, "a_diag must match SheathField");
    immutable double b_effective = b_cell - u_coeff*q_star;
    assert(abs(b_effective - sf_b_rhs) < 1.0e-12,
           "with the node frozen, b_cell - Uc*q must equal SheathField's b_rhs");

    // and the row residual must be the physical -S*J_lin
    immutable double phi_probe = 2.4;
    immutable double J_lin = J0 + Jp*((phi_probe - q_star) - (phi_star - q_star));
    immutable double resid = a_diag*phi_probe + u_coeff*q_star - b_cell;
    assert(abs(resid - (-S*J_lin)) < 1.0e-12, "cell row must represent -S*J_lin");
}

// Identity 2: the NODE row must be Kirchhoff's law written with the SAME sign
// convention assemble_L_and_c uses for the network (L += +1/R, c += +V/R).
// Checking only that the cell and node rows are "equal and opposite" is NOT
// enough -- that is satisfied by both sign choices, and the wrong one silently
// negates half the row relative to the network stamp.
unittest {
    immutable double S = 2.0, J0 = 0.5, Jp = 1.25, phi_star = 1.0, q_star = 7.0;
    immutable double R = 4.0, Vsup = 33.0;
    double a_diag, u_coeff, b_cell, wt_coeff, l_diag, c_node;
    sheathFaceStamp(S, J0, Jp, phi_star, q_star,
                    a_diag, u_coeff, b_cell, wt_coeff, l_diag, c_node);

    // network stamp, exactly as assemble_L_and_c produces it
    auto ck = new ExternalCircuit();
    ck.addNode(q_star); ck.addSupplyLeg(0, R, Vsup);
    double[] Lnet, cnet; ck.assemble_L_and_c(Lnet, cnet);

    // Full node row: Wt*phi + L*q = c. Its residual must equal
    // S*J_lin + (q - Vsup)/R, which is Kirchhoff at this electrode.
    immutable double phi_probe = 1.9, q_probe = 6.2;
    immutable double L_total = l_diag + Lnet[0];
    immutable double c_total = c_node + cnet[0];
    immutable double resid = wt_coeff*phi_probe + L_total*q_probe - c_total;
    immutable double J_lin = J0 + Jp*((phi_probe - q_probe) - (phi_star - q_star));
    immutable double expect = S*J_lin + (q_probe - Vsup)/R;
    assert(abs(resid - expect) < 1.0e-12,
           format("node row must be Kirchhoff in the network's sign convention: "
                  ~ "%g vs %g", resid, expect));
}

version(unittest) {
    // Dense reference: assemble and solve the full (N+K)x(N+K) augmented system.
    private void denseAugmentedSolve(const double[] A0, const double[][] Uc,
                                     const double[][] Wt, const double[] L,
                                     const double[] b0, const double[] c,
                                     size_t N, size_t K,
                                     ref double[] phi, ref double[] q)
    {
        import nm.bbla : Matrix, gaussJordanElimination;
        auto M = new Matrix!double(N+K, N+K+1);
        foreach (i; 0 .. N+K) foreach (j; 0 .. N+K+1) M[i, j] = 0.0;
        foreach (i; 0 .. N) foreach (j; 0 .. N) M[i, j] = A0[i*N + j];
        foreach (i; 0 .. N) foreach (m; 0 .. K) M[i, N+m] = Uc[m][i];
        foreach (m; 0 .. K) foreach (j; 0 .. N) M[N+m, j] = Wt[m][j];
        foreach (m; 0 .. K) foreach (n; 0 .. K) M[N+m, N+n] = L[m*K + n];
        foreach (i; 0 .. N) M[i, N+K] = b0[i];
        foreach (m; 0 .. K) M[N+m, N+K] = c[m];
        gaussJordanElimination!double(M);
        phi.length = N; q.length = K;
        foreach (i; 0 .. N) phi[i] = M[i, N+K];
        foreach (m; 0 .. K) q[m] = M[N+m, N+K];
    }

    // Dense solve of A0 alone, to stand in for GMRES in schurSolve.
    private double[] denseSolveA0(const double[] A0, size_t N, const(double)[] rhs) {
        import nm.bbla : Matrix, gaussJordanElimination;
        auto M = new Matrix!double(N, N+1);
        foreach (i; 0 .. N) {
            foreach (j; 0 .. N) M[i, j] = A0[i*N + j];
            M[i, N] = rhs[i];
        }
        gaussJordanElimination!double(M);
        auto x = new double[N];
        foreach (i; 0 .. N) x[i] = M[i, N];
        return x;
    }
}

// Identity 3 (the decisive one): Woodbury/Schur must agree with a direct dense
// solve of the full augmented system, on a problem built with the SAME stamp
// function efield.d uses. This is the D-side twin of tools/woodbury/woodbury_check.py.
unittest {
    // A small 1D chain of cells with a Dirichlet anchor at one end and two
    // sheath-coupled electrodes at the other, tied to 2 circuit nodes.
    immutable size_t n = 6, K = 2;
    immutable size_t N = n;
    auto A0 = new double[N*N]; A0[] = 0.0;
    auto b0 = new double[N];   b0[] = 0.0;
    double[][] Uc; Uc.length = K; foreach (m; 0 .. K) { Uc[m] = new double[N]; Uc[m][] = 0.0; }
    double[][] Wt; Wt.length = K; foreach (m; 0 .. K) { Wt[m] = new double[N]; Wt[m][] = 0.0; }
    auto L = new double[K*K]; L[] = 0.0;
    auto c = new double[K];   c[] = 0.0;

    // conduction
    foreach (i; 0 .. N) {
        if (i > 0)   { A0[i*N + i] += 1.3; A0[i*N + i-1] -= 1.3; }
        if (i+1 < N) { A0[i*N + i] += 1.3; A0[i*N + i+1] -= 1.3; }
    }
    // Dirichlet anchor at cell 0
    foreach (j; 0 .. N) A0[0*N + j] = 0.0;
    A0[0] = 1.0; b0[0] = 12.0;

    // two electrode faces on the last two cells, one per node
    immutable double[2] Sf   = [1.1, 0.9];
    immutable double[2] J0f  = [0.4, -0.25];
    immutable double[2] Jpf  = [0.65, 0.9];
    immutable double[2] phis = [2.0, 2.5];
    immutable double[2] qs   = [30.0, -10.0];
    foreach (t; 0 .. 2) {
        size_t k = N-1-t; size_t m = t;
        double ad, uc, bc, wt, ld, cn;
        sheathFaceStamp(Sf[t], J0f[t], Jpf[t], phis[t], qs[t], ad, uc, bc, wt, ld, cn);
        A0[k*N + k] += ad;
        Uc[m][k]    += uc;
        b0[k]       += bc;
        Wt[m][k]    += wt;
        L[m*K + m]  += ld;
        c[m]        += cn;
    }
    // external network: a bus between the nodes and a supply leg on each
    auto ck = new ExternalCircuit();
    ck.addNode(30.0, "n0"); ck.addNode(-10.0, "n1");
    ck.addResistor(0, 1, 0.4);
    ck.addSupplyLeg(0, 1.5, 30.0);
    ck.addSupplyLeg(1, 2.5, -10.0);
    double[] Lnet, cnet;
    ck.assemble_L_and_c(Lnet, cnet);
    foreach (i; 0 .. K*K) L[i] += Lnet[i];
    foreach (i; 0 .. K)   c[i] += cnet[i];

    double[] phi_d, q_d, phi_w, q_w;
    denseAugmentedSolve(A0, Uc, Wt, L, b0, c, N, K, phi_d, q_d);
    auto solver = delegate double[](const(double)[] rhs) { return denseSolveA0(A0, N, rhs); };
    schurSolve(solver, Uc, Wt, L, b0, c, phi_w, q_w);

    foreach (i; 0 .. N)
        assert(abs(phi_d[i] - phi_w[i]) < 1.0e-9,
               format("phi[%d]: dense %g vs woodbury %g", i, phi_d[i], phi_w[i]));
    foreach (m; 0 .. K)
        assert(abs(q_d[m] - q_w[m]) < 1.0e-9,
               format("q[%d]: dense %g vs woodbury %g", m, q_d[m], q_w[m]));
}
