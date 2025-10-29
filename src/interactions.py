################################################################################################
#contains classes to handle formulas for calculation of atom-atom- and light-atom-interactions.#
################################################################################################

from numba import njit, float64, int32
from numba.experimental import jitclass
import scipy.constants as scc
import numpy as np
import src.interaction_wrappers.six_level_wrappers as slw
import src.interaction_wrappers.eighteen_level_wrappers as elw
import src.interaction_wrappers.four_level_wrappers as flw

BOHR_MAGNETON = scc.physical_constants["Bohr magneton"][0]

six_level_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
    ("mu_B", float64),
    ("allowed_transitions", int32[:, :]),
    ("ground_mJ", float64[:]),
    ("excited_mJ", float64[:]),
    ("branch_table", float64[:, :, :])
]

@jitclass(six_level_spec)
class Lithium6LevelInteraction:
    def __init__(self):
        self.number_of_ground_states = 2
        self.number_of_excited_states = 4
        self.mu_B = BOHR_MAGNETON
        
        # Define m_J values explicitly
        self.ground_mJ = np.array([-0.5, +0.5])
        self.excited_mJ = np.array([-1.5, -0.5, +0.5, +1.5])
        
        # Allowed transitions matrix: (ground_state, excited_state, polarization)
        # Polarization: 0 -> σ-, 1 -> π, 2 -> σ+
        self.allowed_transitions = np.array([
            [0, 0, 0],  # Ground 0 (-1/2) -> Excited 0 (-3/2), σ-
            [1, 1, 0],  # Ground 1 (+1/2) -> Excited 1 (-1/2), σ-
            [0, 1, 1],  # Ground 0 (-1/2) -> Excited 1 (-1/2), π
            [1, 2, 1],  # Ground 1 (+1/2) -> Excited 2 (+1/2), π
            [0, 2, 2],  # Ground 0 (-1/2) -> Excited 2 (+1/2), σ+
            [1, 3, 2]   # Ground 1 (+1/2) -> Excited 3 (+3/2), σ+
        ], dtype=np.int32)




        # Build a 3D “branching‐weight” table (excited × ground × pol)
        #    entry = |<g; 1,q | e>|^2 for J'=3/2→J=1/2.
        table = np.zeros((self.number_of_excited_states,
                          self.number_of_ground_states,
                          3), dtype=np.float64)

        # e=0 (mJ'=-3/2) → only (g=0, pol=0) has weight=1
        table[0, 0, 0] = 1.0

        # e=1 (mJ'=-½):
        #    (g=0, pol=1) → weight=1/3;  (g=1, pol=0) → weight=2/3
        table[1, 0, 1] = 1.0 / 3.0
        table[1, 1, 0] = 2.0 / 3.0

        # e=2 (mJ'=+½):
        #    (g=0, pol=2) → weight=2/3;  (g=1, pol=1) → weight=1/3
        table[2, 0, 2] = 2.0 / 3.0
        table[2, 1, 1] = 1.0 / 3.0

        # e=3 (mJ'=+3/2) → only (g=1, pol=2) has weight=1
        table[3, 1, 2] = 1.0

        self.branch_table = table       

    def calculate_rate(self, 
                       saturation_parameters,
                       total_saturation_parameter,
                       natural_linewidth,
                       excitation_rates):
        
        slw._calculate_excitation_rate(saturation_parameters,
                                       total_saturation_parameter,
                                       natural_linewidth,
                                       excitation_rates)
        


    

    
    def calculate_saturation_parameter(self,
                                    polarization: int, 
                                    magnetic_field_strength: float, 
                                    ground_state: float, 
                                    excited_state: float, 
                                    laser_intensity: float, 
                                    natural_linewidth: float, 
                                    saturation_intensity: float, 
                                    effective_transition_frequency: float, 
                                    doppler_shift, 
                                    laser_beam_frequency: float, 
                                    detuning: float):
    
        
        transition_strength = self.branch_table[ground_state][excited_state][polarization]**2
        saturation_parameter = slw._calculate_saturation_parameter(laser_intensity = laser_intensity,
                                                                   natural_linewidth=natural_linewidth,
                                                                   transition_strength=transition_strength,
                                                                   effective_transition_frequency=effective_transition_frequency,
                                                                   doppler_shift = doppler_shift,
                                                                   laser_beam_frequency=laser_beam_frequency,
                                                                   detuning = detuning)
        
        return saturation_parameter

    def calculate_transition_frequency_shift(self, 
                                             polarization: int, 
                                             ground_state: int, 
                                             excited_state: int, 
                                             magnetic_field_strength: float):
        """
        Calculate the Zeeman shift for a given transition.
        
        Parameters:
        - polarization: 0 (σ-), 1 (π), 2 (σ+); lab frame specification.
        - ground_state: index (0 for mJ=-1/2, 1 for mJ=+1/2)
        - excited_state: index (0 for mJ=-3/2, 1 for mJ=-1/2, etc.)
        - magnetic_field_strength: Magnetic field (Tesla). Sign indicates field direction.
        
        Returns:
        - Energy shift (Joules) for the transition.
        """
        # Adjust polarization if the field is anti-parallel to the laser.




        transition_energy_shift = slw._calculate_transition_frequency_shift( 
                                                                            ground_state=ground_state, 
                                                                            excited_state = excited_state, 
                                                                            magnetic_field_strength = magnetic_field_strength, 
                                                                            mu_B = self.mu_B, 
                                                                            ground_mJ = self.ground_mJ, 
                                                                            excited_mJ = self.excited_mJ
                                                                            )
        return transition_energy_shift



    def calculate_branching_ratio(self, 
                                            polarization: int, 
                                            ground_state: int, 
                                            excited_state: int, 
                                            magnetic_field_strength: float):
        return self.branch_table[excited_state, ground_state, polarization]
#######################################################################################################################################
#
#                                               18-Level-Code from Julia
#
#######################################################################################################################################
            

eighteen_level_spec = [("number_of_ground_states", int32),
                       ("number_of_excited_states", int32)]

@jitclass(eighteen_level_spec)
class Lithium18LevelInteraction:

    def __init__(self):

        self.number_of_ground_states = 6
        self.number_of_excited_states = 12


    
    def calculate_transition_frequency_shift(self, 
                                             polarization: int, 
                                             ground_state: int, 
                                             excited_state: int, 
                                             magnetic_field_strength: float):
        """
        Calculate the Zeeman shift for a given transition.
        
        Parameters:
        - polarization: 0 (σ-), 1 (π), 2 (σ+); lab frame specification.
        - ground_state: index (0 for mJ=-1/2, 1 for mJ=+1/2)
        - excited_state: index (0 for mJ=-3/2, 1 for mJ=-1/2, etc.)
        - magnetic_field_strength: Magnetic field (Tesla). Sign indicates field direction.
        
        Returns:
        - Energy shift (Joules) for the transition.
        """

        transition_frequency_shift = elw.calculate_transition_frequency_shift(ground_state, excited_state, polarization, magnetic_field_strength)

        return transition_frequency_shift


    def calculate_rate(self, 
                       saturation_parameters,
                       total_saturation_parameter,
                       natural_linewidth,
                       excitation_rates):
        
        elw._calculate_transition_rate(saturation_parameters,
                                       total_saturation_parameter,
                                       natural_linewidth,
                                       excitation_rates)
    

    def calculate_saturation_parameter(self,
                                       polarization: int, 
                                       magnetic_field_strength: float, 
                                       ground_state: float, 
                                       excited_state: float, 
                                       laser_intensity: float, 
                                       natural_linewidth: float, 
                                       saturation_intensity: float, 
                                       effective_transition_frequency: float, 
                                       doppler_shift, 
                                       laser_beam_frequency: float, 
                                       detuning: float):
        
        transition_strength = elw.calculate_transition_strength(GS = ground_state, ES = excited_state, pol = polarization, B = magnetic_field_strength)

        saturation_parameter = elw._calculate_saturation_parameter(effective_transition_frequency=effective_transition_frequency,
                                                         doppler_shift=doppler_shift,
                                                         laser_beam_frequency=laser_beam_frequency,
                                                         detuning = detuning,
                                                         transition_strength= transition_strength,
                                                         laser_intensity = laser_intensity,
                                                         natural_linewidth = natural_linewidth
                                                         )
        
        return saturation_parameter



    def calculate_branching_ratio(self, 
                                  polarization: int, 
                                  ground_state: int, 
                                  excited_state: int, 
                                  magnetic_field_strength: float):
        
        return elw.calculate_transition_strength(ground_state, excited_state, polarization, magnetic_field_strength)

four_level_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
    ("mu_B", float64),
    ("ground_mJ", float64[:]),
    ("excited_mJ", float64[:]),
    ("allowed_transitions", int32[:, :])
]

@jitclass(four_level_spec)
class Lithium4LevelInteraction:


    def __init__(self):
        self.number_of_ground_states = 1
        self.number_of_excited_states = 3

        self.mu_B = BOHR_MAGNETON

        self.ground_mJ = np.array([0], dtype = np.float64)
        self.excited_mJ = np.array([-1, 0, 1], dtype = np.float64)

        self.allowed_transitions = np.array([[1,0,0],
                                            [1,1,1],
                                            [1,2,2]], dtype=np.int32) #ground state, excited state, polarization


    def calculate_rate(self, 
                       saturation_parameters,
                       total_saturation_parameter,
                       natural_linewidth,
                       excitation_rates):
        
        flw._calculate_excitation_rate(saturation_parameters = saturation_parameters,
                                                          total_saturation_parameter=total_saturation_parameter,
                                                          natural_linewidth=natural_linewidth,
                                                          excitation_rates=excitation_rates)
        return 
    


    def calculate_saturation_parameter(self,
                                    polarization: int, 
                                    magnetic_field_strength: float, 
                                    ground_state: float, 
                                    excited_state: float, 
                                    laser_intensity: float, 
                                    natural_linewidth: float, 
                                    saturation_intensity: float, 
                                    effective_transition_frequency: float, 
                                    doppler_shift, 
                                    laser_beam_frequency: float, 
                                    detuning: float):
    
        
        if [ground_state, excited_state, polarization] in self.allowed_transitions:
            transition_strength = 1
            saturation_parameter = flw._calculate_saturation_parameter(laser_intensity = laser_intensity,
                                                                    natural_linewidth=natural_linewidth,
                                                                    transition_strength=transition_strength,
                                                                    effective_transition_frequency=effective_transition_frequency,
                                                                    doppler_shift = doppler_shift,
                                                                    laser_beam_frequency=laser_beam_frequency,
                                                                    detuning = detuning)
        else:
            saturation_parameter = 0
        
        return saturation_parameter
    
    def calculate_branching_ratio(self, ground_state: int,
                                        excited_state: int,
                                        polarization: int,
                                        magnetic_field_strength: float):
        
        return 1
                                        

    def calculate_transition_frequency_shift(self, 
                                            polarization: int, 
                                            ground_state: int, 
                                            excited_state: int, 
                                            magnetic_field_strength: float):
        """
        Calculate the Zeeman shift for a given transition.
        
        Parameters:
        - polarization: 0 (σ-), 1 (π), 2 (σ+); lab frame specification.
        - ground_state: index (0 for mJ=-1/2, 1 for mJ=+1/2)
        - excited_state: index (0 for mJ=-3/2, 1 for mJ=-1/2, etc.)
        - magnetic_field_strength: Magnetic field (Tesla). Sign indicates field direction.
        
        Returns:
        - Energy shift (Joules) for the transition.
        """
        
        transition_frequency_shift = flw._calculate_transition_frequency_shift( 
                                                                            ground_state=ground_state, 
                                                                            excited_state = excited_state, 
                                                                            magnetic_field_strength = magnetic_field_strength, 
                                                                            mu_B = self.mu_B, 
                                                                            ground_mJ = self.ground_mJ, 
                                                                            excited_mJ = self.excited_mJ
                                                                            )
        return transition_frequency_shift


