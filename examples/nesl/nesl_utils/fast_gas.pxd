cimport cython
cimport numpy as cnp

cdef extern from "gas.h":
    int cwrap_gas_init() noexcept nogil

    int gas_model_new(char* file_name) noexcept nogil
    int gas_model_n_species(int gm_i) noexcept nogil
    int gas_model_n_modes(int gm_i) noexcept nogil
    int gas_model_species_name(int gm_i, int isp, char* name, int n) noexcept nogil
    int gas_model_mol_masses(int gm_i, double* mm) noexcept nogil
    int gas_model_gas_state_update_thermo_from_rhou(int gm_i, int gs_i) noexcept nogil
    int gas_model_gas_state_update_thermo_from_rhoT(int gm_i, int gs_i) noexcept nogil
    int gas_model_gas_state_update_thermo_from_pT(int gm_i, int gs_i) noexcept nogil
    int gas_model_gas_state_update_sound_speed(int gm_i, int gs_i) noexcept nogil
    int gas_model_gas_state_update_trans_coeffs(int gm_i, int gs_i) noexcept nogil
    int gas_model_gas_state_binary_diffusion_coefficients(int gm_i, int gs_i, double* dij) noexcept nogil
    int gas_model_gas_state_enthalpy_isp(int gm_i, int gs_i, int isp, double* result) noexcept nogil
    int gas_model_gas_state_enthalpy_isp_in_mode(int gm_i, int gs_i, int isp, int imode, double* result) noexcept nogil
    int gas_model_gas_state_get_molef(int gm_i, int gs_i, double* molef) noexcept nogil
    int gas_model_gas_state_Cp(int gm_i, int gs_i, double* result) noexcept nogil

    int gas_state_new(int gm_i) noexcept nogil
    int gas_state_copy_values(int gs_to_i, int gs_from_i) noexcept nogil
    int gas_state_set_scalar_field(int gs_i, char* field_name, double value) noexcept nogil
    int gas_state_get_scalar_field(int gs_i, char* field_name, double* value) noexcept nogil
    
    int gas_state_set_array_field(int gs_i, char* field_name, double* values, int n) noexcept nogil
    int gas_state_get_array_field(int gs_i, char* field_name, double* values, int n) noexcept nogil
    
    int thermochemical_reactor_new(int gm_i, char* filename1, char* filename2) noexcept nogil
    int thermochemical_reactor_gas_state_update(int cr_i, int gs_i, double t_interval, double* dt_suggest) noexcept nogil

    int gasflow_normal_shock(int state1_id, double vs, int state2_id, int gm_id, double* results, double rho_tol, double T_tol)

cdef class GasModel:
    cdef readonly int _modelId
    cdef readonly int numSpecies
    cdef readonly int numModes
    cdef readonly list nameSpecies
    cdef readonly double[:] molMasses

cdef class GasState:
    cdef readonly int _modelId
    cdef readonly int _stateId
    cdef readonly int numSpecies
    cdef readonly int numModes
   
    cdef public double rho
    cdef public double p
    cdef public double T 
    cdef public double u
    cdef public double[:,:] binaryCoef
    
    cdef readonly double a
    cdef readonly double k
    cdef readonly double mu
    
    cdef public double[:] massf
    cdef public double[:] T_modes
    cdef public double[:] u_modes
    cdef public double[:] k_modes
    cdef public double[:] molef
        
    cpdef inline void push(self) noexcept nogil
        
    cpdef inline void pull(self) noexcept nogil
 
    cpdef inline int update_thermo_from_rhou(self) noexcept nogil
    
    cpdef inline void update_thermo_from_pT(self) noexcept nogil

    cpdef inline void update_thermo_from_rhoT(self) noexcept nogil

    cpdef inline void update_trans_coeffs(self) noexcept nogil
        
    cpdef inline void update_sound_speed(self) noexcept nogil
        
    cpdef inline void copy_values(self, int gasStateId) noexcept nogil

    cpdef inline double[:,:] binary_diffusion_coefficients(self) noexcept nogil

    cpdef inline double enthalpy_isp(self, int isp) noexcept nogil

    cpdef inline double enthalpy_isp_in_mode(self, int isp, int imode) noexcept nogil

    cpdef inline double[:] get_molef(self) noexcept nogil

    cpdef inline double get_Cp(self) noexcept nogil

cdef class GasFlow:
    cdef readonly int _modelId
    cdef readonly cnp.ndarray nsResults

    cpdef inline cnp.ndarray normal_shock(self, GasState state1, double vs, GasState state2, double rho_tol=1.0e-6, double T_tol=0.1) noexcept

cdef class ThermochemicalReactor:
    cdef readonly int _reactorId
    cdef readonly int status

    cpdef inline double update_state(self, GasState gstate, double tInterval, double dtSuggest) noexcept nogil



