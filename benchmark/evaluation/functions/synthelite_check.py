from functions.utils import flip_reactions, get_route_len
from functions.code_3597 import main as has_heterocycle
from functions.code_11207 import main as has_late_suzuki
from functions.code_7800 import main as has_amide_coupling_then_cross_coupling
from functions.code_7800_amine import main as has_amination_then_cross_coupling
from functions.code_9846 import main as has_late_functionalization
from functions.code_9266 import main as has_late_halogenation
from functions.code_219 import main as has_late_amide_coupling_with_heterocycle
from functions.code_5295 import main as has_late_heterocycle
from functions.code_2866 import main as has_preserved_group
from functions.code_1971 import main as has_late_cc
from functions.code_16 import main as has_late_cf3
from functions.code_11130 import main as has_preserved_methylsulfonyl
from functions.code_7969 import main as has_core_elaboration
from functions.has_aromatic_floride import main as has_aromatic_floride
from functions.has_cut_between_rings import main as has_cut_between_rings
from functions.has_late_methylsulfonyl import main as has_late_methylsulfonyl
from functions.has_late_aldol import main as has_late_aldol

import json
import importlib

def check_piperidine_and_oxoisoindolinone_break(route):
    check, metadata = has_heterocycle(route)
    has_dionering = check and 'piperidine-2,6-dione' in metadata['atomic_checks']['ring_systems']
    has_piperidine = check and 'piperidine' in metadata['atomic_checks']['ring_systems']
    has_oxoisoindolinone = check and 'oxoisoindolinone' in metadata['atomic_checks']['ring_systems']
    return has_piperidine and has_oxoisoindolinone and not has_dionering

def check_dionering_and_oxoisoindolinone_break(route):
    check, metadata = has_heterocycle(route)
    has_dionering = check and 'piperidine-2,6-dione' in metadata['atomic_checks']['ring_systems']
    has_piperidine = check and 'piperidine' in metadata['atomic_checks']['ring_systems']
    has_oxoisoindolinone = check and 'oxoisoindolinone' in metadata['atomic_checks']['ring_systems']
    return has_dionering and has_oxoisoindolinone and not has_piperidine

def check_oxoisoindolinone_break(route):
    check, metadata = has_heterocycle(route)
    has_dionering = check and 'piperidine-2,6-dione' in metadata['atomic_checks']['ring_systems']
    has_piperidine = check and 'piperidine' in metadata['atomic_checks']['ring_systems']
    has_oxoisoindolinone = check and 'oxoisoindolinone' in metadata['atomic_checks']['ring_systems']
    return not has_dionering and has_oxoisoindolinone and not has_piperidine

def check_early_imidazole(route):
    check, metadata = has_heterocycle(route)
    route_len = get_route_len(route)
    has_cycle = check and 'imidazole' in metadata['atomic_checks']['ring_systems']
    if not has_cycle:
        return True
    # return metadata.get('at_depth', -1) >= route_len-1
 
    return bool(set([route_len-1, route_len]) & set(metadata.get('at_depths', [])))

def check_late_imidazole(route):
    check, metadata = has_heterocycle(route)
    if not check:
        return False
    if not bool(set([1, 2]) & set(metadata.get('at_depths', []))):
        return False
    return 'imidazole' in metadata['atomic_checks']['ring_systems'] or 'imidazole' in ','.join(metadata['atomic_checks']['named_reactions'])


def check_early_pyrimidine(route):
    check, metadata = has_heterocycle(route)
    route_len = get_route_len(route)
    has_cycle = check and 'pyrimidine' in metadata['atomic_checks']['ring_systems']
    if not has_cycle:
        return True
    # return metadata.get('at_depth', -1) >= route_len-1
    return bool(set([route_len-1, route_len]) & set(metadata.get('at_depths', [])))

def check_no_ring_formation(route):
    check, _ = has_heterocycle(route)
    return not check

def _has_pyrazole(route):
    check, metadata = has_heterocycle(route)
    return check and 'pyrazole' in metadata['atomic_checks']['ring_systems']

def check_late_pyrazole(route):
    check, metadata = has_heterocycle(route)
    return check and 'pyrazole' in metadata['atomic_checks']['ring_systems'] \
        and bool(set([1, 2]) & set(metadata.get('at_depths', [])))
        # and metadata.get('at_depth', 100) <=2

def check_early_pyrazole(route):
    check, metadata = has_heterocycle(route)
    route_len = get_route_len(route)
    has_cycle = check and 'pyrazole' in metadata['atomic_checks']['ring_systems']
    if not has_cycle:
        return True # If no pyrazole, return True for early pyrazole check
    # return metadata.get('at_depth', -1) >= route_len-1
    return bool(set([route_len-1, route_len]) & set(metadata.get('at_depths', [])))

def check_pyrazole_and_late_suzuki(route):
    if not _has_pyrazole(route):
        return False
    check, _ = has_late_suzuki(route)
    return check

def check_heterocycle_amination_suzuki(route):
    hetcycle_check, hetcycle_metadata = has_heterocycle(route)
    if not hetcycle_check:
        return False
    
    amination_then_cc, _ = has_amination_then_cross_coupling(route) 
    if not amination_then_cc:
        return False
    
    return hetcycle_metadata['at_depths'][0] >=3 # check if heterocycle is formed before amide coupling and cross coupling
    
def check_late_reductive_amination(route):
    check, metadata = has_late_functionalization(route)
    # is_reductive_amination = 'reductive amination' in metadata['atomic_checks'].get('named_reactions', '')
    if not check:
        return False
    reaction_names = ' '.join(metadata['atomic_checks'].get('named_reactions', [])).lower()
    carbon_fgs = ' '.join(metadata['atomic_checks'].get('functional_groups', [])).lower()
    is_reductive_amination = 'amine' in reaction_names or 'amination' in reaction_names or 'aldehyde' in carbon_fgs or 'ketone' in carbon_fgs
    return is_reductive_amination

def check_suzuki_pyridine(route):
    check, metadata = has_late_suzuki(route)
    has_pyridine = 'pyridine' in metadata['atomic_checks'].get('ring_systems', [])
    return check and has_pyridine

def check_early_pyrrole(route):
    check, metadata = has_heterocycle(route)
    route_len = get_route_len(route)
    has_cycle = check and 'pyrrole' in metadata['atomic_checks']['ring_systems']
    if not has_cycle:
        return True
    # return metadata.get('at_depth', -1) >= route_len-1
    return bool(set([route_len-1, route_len]) & set(metadata.get('at_depths', [])))

def check_late_F(route):
    check, metadata = has_aromatic_floride(route)
    if not check:
        return False
    return max(metadata['at_depths']) <= 2
    # check, _ = has_late_cf3(route)
    # return check

def check_late_amide_coupling(route):
    check, _ = has_late_amide_coupling_with_heterocycle(route)
    return check

def check_late_piperazine(route):
    check, metadata = has_heterocycle(route)
    if not check:
        return False
    if 'piperazine' not in metadata['atomic_checks']['ring_systems']:
        return False
    return max(metadata['at_depths']) <= 2

def check_preserved_chloro_benzene(route):
    check, _ = has_preserved_group(route)
    return check

def check_late_trifluoromethylation(route):
    check = has_late_cf3(route)
    return check

def check_preserved_methylsulfonyl(route):
    check, _ = has_preserved_methylsulfonyl(route)
    return check

def check_late_methylsulfonyl(route):
    check = has_late_methylsulfonyl(route)
    return check

def check_disconnection_with_indole(route):
    check, metadata = has_cut_between_rings(route)
    # if not check:
    #     return False
    
    # has_indole = False
    # has_other_ring = False
    for reactant_rings in metadata["atomic_checks"]["ring_systems"][:2]:
        if len(reactant_rings) <= 2 and "indole" in reactant_rings:
            return True
        
    return False
        
def check_disconnection_between_piperidine_and_piperazine(route):
    check, metadata = has_cut_between_rings(route)

    for reactant_rings in metadata["atomic_checks"]["ring_systems"][:2]:
        # if len(reactant_rings) == 1 and reactant_rings[0] == "piperidine":
        #     return True
        if "piperazine" in reactant_rings and "thiophene" in reactant_rings:
            return True
        
    return False

def check_disconnection_between_piperidines(route):
    _, metadata = has_cut_between_rings(route)
    
    n_reactants_with_piperidine = 0
    for reactant_rings in metadata["atomic_checks"]["ring_systems"][:2]:
        if "piperidine" in reactant_rings:
            n_reactants_with_piperidine += 1
            
    return n_reactants_with_piperidine == 2

def check_disconnection_with_thiophene(route):
    check, metadata = has_cut_between_rings(route)

    for reactant_rings in metadata["atomic_checks"]["ring_systems"][:2]:
        if len(reactant_rings) == 1 and reactant_rings[0] == "thiophene":
            return True
        
    return False

def check_late_aldol(route):
    check, metadata = has_late_aldol(route)
    if not check:
        return False
    return max(metadata['at_depths']) <= 2

def check_heterocycle_then_snar(route):
    check, _ = has_core_elaboration(route)
    return check

def get_synthelite_check_func(name: str):
    func_name = name
    try:
        func = globals()[func_name]
        if callable(func):
            return func
        else:
            raise ValueError(f"{func_name} is not callable")
    except KeyError:
        raise ValueError(f"Function {func_name} not found")
    
def apply_func_to_route(func_name: str, route):
    func = get_synthelite_check_func(func_name)
    route = flip_reactions(route)
    return func(route)