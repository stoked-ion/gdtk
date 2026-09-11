# type: ignore
cimport cython
import numpy as np
cimport numpy as cnp
cimport fast_gas
cnp.import_array()
cwrap_gas_init()  

@cython.final
cdef class GasModel:
    def __init__(self, filename):
        self._modelId = gas_model_new(bytes(filename, 'utf-8'))
        self.numSpecies = gas_model_n_species(self._modelId)
        self.numModes = gas_model_n_modes(self._modelId)
        
        cdef char buf[32]
        self.nameSpecies = []
        for i in range(self.numSpecies):
            gas_model_species_name(self._modelId, i, buf, 32)
            self.nameSpecies.append(buf.decode("utf-8"))

        self.molMasses = np.zeros(self.numSpecies, dtype=np.double)
        gas_model_mol_masses(self._modelId, &(cython.cast(double[:],self.molMasses)[0]))

@cython.final
cdef class GasState:
    def __init__(self, gasModel):
        self._modelId = gasModel._modelId
        self._stateId = gas_state_new(gasModel._modelId)      
        self.numSpecies = gas_model_n_species(gasModel._modelId)
        self.numModes = gas_model_n_modes(gasModel._modelId)
        self.T_modes = np.zeros(self.numModes, dtype=np.double)
        self.u_modes = np.zeros(self.numModes, dtype=np.double)
        self.k_modes = np.zeros(self.numModes, dtype=np.double)
        self.massf = np.zeros(self.numSpecies, dtype=np.double)
        self.molef = np.zeros(self.numSpecies, dtype=np.double)
        self.binaryCoef = np.zeros((self.numSpecies, self.numSpecies), dtype=np.double)
        
    @cython.boundscheck(False)
    cpdef inline void push(self) noexcept nogil:
        gas_state_set_scalar_field(self._stateId, "T", self.T)
        gas_state_set_scalar_field(self._stateId, "p", self.p)
        gas_state_set_scalar_field(self._stateId, "u", self.u)
        gas_state_set_scalar_field(self._stateId, "rho", self.rho)
        gas_state_set_array_field(self._stateId, "massf", &(cython.cast(double[:],self.massf)[0]), self.numSpecies)
        gas_state_set_array_field(self._stateId, "T_modes", &(cython.cast(double[:],self.T_modes)[0]), self.numModes)
        gas_state_set_array_field(self._stateId, "u_modes", &(cython.cast(double[:],self.u_modes)[0]), self.numModes)
        
    @cython.boundscheck(False)
    cpdef inline void pull(self) noexcept nogil:
        gas_state_get_scalar_field(self._stateId, "T", &(self.T))
        gas_state_get_scalar_field(self._stateId, "p", &(self.p))
        gas_state_get_scalar_field(self._stateId, "u", &(self.u))
        gas_state_get_scalar_field(self._stateId, "mu", &(self.mu))
        gas_state_get_scalar_field(self._stateId, "k", &(self.k))
        gas_state_get_scalar_field(self._stateId, "a", &(self.a))
        gas_state_get_scalar_field(self._stateId, "rho", &(self.rho))
        gas_state_get_array_field(self._stateId, "massf", &(cython.cast(double[:],self.massf)[0]), self.numSpecies)
        gas_state_get_array_field(self._stateId, "T_modes", &(cython.cast(double[:],self.T_modes)[0]), self.numModes)
        gas_state_get_array_field(self._stateId, "u_modes", &(cython.cast(double[:],self.u_modes)[0]), self.numModes) 
        gas_state_get_array_field(self._stateId, "k_modes", &(cython.cast(double[:],self.k_modes)[0]), self.numModes) 
    
    cpdef inline int update_thermo_from_rhou(self) noexcept nogil:  # Returns 0 on success, -1 if the gas_cwrap.d threw non-finite / invalid state.
        return gas_model_gas_state_update_thermo_from_rhou(self._modelId, self._stateId) 
    
    cpdef inline void update_thermo_from_pT(self) noexcept nogil:
        gas_model_gas_state_update_thermo_from_pT(self._modelId, self._stateId)
    
    cpdef inline void update_thermo_from_rhoT(self) noexcept nogil:
        gas_model_gas_state_update_thermo_from_rhoT(self._modelId, self._stateId)

    cpdef inline void update_trans_coeffs(self) noexcept nogil:
        gas_model_gas_state_update_trans_coeffs(self._modelId, self._stateId)
        
    cpdef inline void update_sound_speed(self) noexcept nogil:
        gas_model_gas_state_update_sound_speed(self._modelId, self._stateId)
        
    cpdef inline void copy_values(self, int gasStateId) noexcept nogil:
        gas_state_copy_values(self._stateId, gasStateId)
    
    cpdef inline double[:,:] binary_diffusion_coefficients(self) noexcept nogil:
        gas_model_gas_state_binary_diffusion_coefficients(self._modelId, self._stateId, &(self.binaryCoef[0,0]))
        return self.binaryCoef

    cpdef inline double enthalpy_isp(self, int isp) noexcept nogil:
        cdef double enthalpyIsp = 0.0
        gas_model_gas_state_enthalpy_isp(self._modelId, self._stateId, isp, &enthalpyIsp)
        return enthalpyIsp

    cpdef inline double enthalpy_isp_in_mode(self, int isp, int imode) noexcept nogil:
        cdef double enthalpyIspInMode = 0.0
        gas_model_gas_state_enthalpy_isp_in_mode(self._modelId, self._stateId, isp, imode, &enthalpyIspInMode)
        return enthalpyIspInMode

    cpdef inline double[:] get_molef(self) noexcept nogil:
        gas_model_gas_state_get_molef(self._modelId, self._stateId, &(cython.cast(double[:],self.molef)[0]))
        return self.molef

    cpdef inline double get_Cp(self) noexcept nogil:
        cdef double Cp = 0.0
        gas_model_gas_state_Cp(self._modelId, self._stateId, &Cp)
        return Cp

@cython.final
cdef class GasFlow:
    def __init__(self, gasModel):
        self._modelId = gasModel._modelId
        self.nsResults = np.zeros(2, dtype=np.double)

    cpdef inline cnp.ndarray normal_shock(self, GasState state1, double vs, GasState state2, double rho_tol=1.0e-6, double T_tol=0.1) noexcept:    
        gasflow_normal_shock(state1._stateId, vs, state2._stateId, self._modelId, &(cython.cast(double[:], self.nsResults)[0]), rho_tol, T_tol)
        state2.pull()
        return self.nsResults

@cython.final
cdef class ThermochemicalReactor:    
    def __init__(self, gasModel, filename1, filename2=""):
        self._reactorId = thermochemical_reactor_new(gasModel._modelId, bytes(filename1, 'utf-8'), bytes(filename2, 'utf-8'))
        self.status = 0

    cpdef inline double update_state(self, GasState gstate, double tInterval, double dtSuggest) noexcept nogil:  # Status is 0 on success, -1 if the reactor threw, checked by the caller.
        self.status = thermochemical_reactor_gas_state_update(self._reactorId, gstate._stateId, tInterval, &dtSuggest)
        return dtSuggest


