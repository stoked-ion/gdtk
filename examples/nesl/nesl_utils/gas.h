int cwrap_gas_init();

int gas_model_new(char* file_name);
int gas_model_type_str(int gm_i, char* dest_str, int n);
int gas_model_n_species(int gm_i);
int gas_model_n_modes(int gm_i);
int gas_model_species_name(int gm_i, int isp, char* name, int n);
int gas_model_mol_masses(int gm_i, double* mm);

int gas_state_new(int gm_i);
int gas_state_set_scalar_field(int gs_i, char* field_name, double value);
int gas_state_get_scalar_field(int gs_i, char* field_name, double* value);
int gas_state_get_thermo_scalars(int gs_i, double* values);
int gas_state_set_array_field(int gs_i, char* field_name, double* values, int n);
int gas_state_get_array_field(int gs_i, char* field_name, double* values, int n);
int gas_state_get_ceaSavedData_field(int gs_i, char* field_name, double* value);
int gas_state_get_ceaSavedData_massf(int gs_i, char* species_name, double* value);
int gas_state_get_ceaSavedData_species_names(int gs_i, char* dest_str, int n);
int gas_state_copy_values(int gs_to_i, int gs_from_i);
int gas_state_get_saved_species_names(int gs_i, char* dest_str, int n);
int gas_state_get_saved_massf(int gs_i, int isp, double* value);

int gas_model_gas_state_update_thermo_from_pT(int gm_i, int gs_i);
int gas_model_gas_state_update_thermo_from_rhou(int gm_i, int gs_i);
int gas_model_gas_state_update_thermo_from_rhoT(int gm_i, int gs_i);
int gas_model_gas_state_update_thermo_from_rhop(int gm_i, int gs_i);
int gas_model_gas_state_update_thermo_from_ps(int gm_i, int gs_i, double s);
int gas_model_gas_state_update_thermo_from_hs(int gm_i, int gs_i, double h, double s);
int gas_model_gas_state_update_sound_speed(int gm_i, int gs_i);
int gas_model_gas_state_update_trans_coeffs(int gm_i, int gs_i);

int gas_model_gas_state_Cv(int gm_i, int gs_i, double* result);
int gas_model_gas_state_Cp(int gm_i, int gs_i, double* result);
int gas_model_gas_state_dpdrho_const_T(int gm_i, int gs_i, double* result);
int gas_model_gas_state_R(int gm_i, int gs_i, double* result);
int gas_model_gas_state_gamma(int gm_i, int gs_i, double* result);
int gas_model_gas_state_Prandtl(int gm_i, int gs_i, double* result);
int gas_model_gas_state_internal_energy(int gm_i, int gs_i, double* result);
int gas_model_gas_state_enthalpy(int gm_i, int gs_i, double* result);
int gas_model_gas_state_entropy(int gm_i, int gs_i, double* result);
int gas_model_gas_state_molecular_mass(int gm_i, int gs_i, double* result);
int gas_model_gas_state_binary_diffusion_coefficients(int gm_i, int gs_i, double* dij);

int gas_model_gas_state_enthalpy_isp(int gm_i, int gs_i, int isp, double* result);
int gas_model_gas_state_enthalpy_isp_in_mode(int gm_i, int gs_i, int isp, int imode, double* result);
int gas_model_gas_state_entropy_isp(int gm_i, int gs_i, int isp, double* result);
int gas_model_gas_state_gibbs_free_energy_isp(int gm_i, int gs_i, int isp, double* result);

int gas_model_massf2molef(int gm_i, double* massf, double* molef);
int gas_model_molef2massf(int gm_i, double* molef, double* massf);
int gas_model_gas_state_get_molef(int gm_i, int gs_i, double* molef);
int gas_model_gas_state_get_conc(int gm_i, int gs_i, double* conc);

int thermochemical_reactor_new(int gm_i, char* filename1, char* filename2);
int thermochemical_reactor_gas_state_update(int cr_i, int gs_i, double t_interval,
											double* dt_suggest);
int thermochemical_reactor_eval_source_terms(int cr_i, int gm_i, int gs_i, int nsp, int nmodes, double* source);

int reaction_mechanism_new(int gm_i, char* filename);
int reaction_mechanism_n_reactions(int rm_i);
int reaction_mechanism_tickrates(int rm_i, int gm_i, int gs_i, double* forwardrates, double* backwardrates);

int gasflow_shock_ideal(int state1_id, double vs, int state2_id, int gm_id,
						double* results);
int gasflow_normal_shock(int state1_id, double vs, int state2_id, int gm_id,
						 double* results, double rho_tol, double T_tol);
int gasflow_normal_shock_1(int state1_id, double vs, int state2_id, int gm_id,
						   double* results, double p_tol, double T_tol);
int gasflow_normal_shock_p2p1(int state1_id, double p2p1, int state2_id, int gm_id,
							  double* results);
int gasflow_reflected_shock(int state2_id, double vg, int state5_id, int gm_id,
							double* results);

int gasflow_expand_from_stagnation(int state0_id, double p_over_p0, int state1_id,
								   int gm_id, double* results);
int gasflow_expand_to_mach(int state0_id, double mach, int state1_id,
						   int gm_id, double* results);
int gasflow_total_condition(int state1_id, double v1, int state0_id, int gm_id);
int gasflow_pitot_condition(int state1_id, double v1, int state2pitot_id, int gm_id);
int gasflow_steady_flow_with_area_change(int state1_id, double v1, double a2_over_a1,
										 int state2_id, int gm_id, double tol, double p2p1_min,
										 double* results);

int gasflow_finite_wave_dp(int state1_id, double v1, char* characteristic, double p2,
						   int state2_id, int gm_id, int steps, double* results);
int gasflow_finite_wave_dv(int state1_id, double v1, char* characteristic, double v2_target,
						   int state2_id, int gm_id, int steps, double Tmin, double* results);

int gasflow_osher_riemann(int stateL_id, int stateR_id, double velL, double velR,
						  int stateLstar_id, int stateRstar_id,
						  int stateX0_id, int gm_id, double* results);
int gasflow_osher_flux(int stateL_id, int stateR_id, double velL, double velR,
					   int gm_id, double* results);
int gasflow_roe_flux(int stateL_id, int stateR_id, double velL, double velR,
					 int gm_id, double* results);

int gasflow_lrivp(int stateL_id, int stateR_id, double velL, double velR,
				  int gmL_id, int gmR_id, double* wstar, double* pstar);
int gasflow_piston_at_left(int stateR_id, double velR, int gm_id,
						   double wstar, double* pstar);
int gasflow_piston_at_right(int stateL_id, double velL, int gm_id,
							double wstar, double* pstar);

int gasflow_theta_oblique(int state1_id, double v1, double beta,
						  int state2_id, int gm_id, double* results);
int gasflow_beta_oblique(int state1_id, double v1, double theta,
						 int gm_id, double* results);

int gasflow_theta_cone(int state1_id, double v1, double beta,
					   int state_c_id, int gm_id, double dtheta, double* results);
int gasflow_beta_cone(int state1_id, double v1, double theta,
					  int gm_id, double dtheta, double* results);
