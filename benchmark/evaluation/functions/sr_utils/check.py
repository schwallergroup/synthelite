import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import itertools  # <-- Make sure to import itertools

class Check:
    def __init__(self, fg_dict, reaction_dict, ring_dict):
        self.fg_dict = fg_dict
        self.reaction_class_dict = reaction_dict
        self.ring_dict = ring_dict

    def check_ring(self, name, smi):
        ring_smiles = self.ring_dict[name]
        if ring_smiles == []:
            print(f"No smiles found for {name}")
            return False
        # Create the molecule object once to avoid repeated parsing
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Could not parse SMILES: {smi}")
            return False
        for ring_smi in ring_smiles:
            patt = Chem.MolFromSmiles(ring_smi)
            if patt is None:
                continue
            if mol.HasSubstructMatch(patt):
                return True
        return False

    def get_ring_smiles(self, name):
        ring_smiles = self.ring_dict[name]
        if not ring_smiles:
            print(f"No smiles found for {name}")
            return []
        return ring_smiles[0]

    def get_ring_atom_indices(self, ring_name, query_mol_smiles):
        """
        Find the atom indices in the query molecule where a specific ring pattern matches.

        Parameters:
        ring_name (str): The name of the ring to search for
        query_mol_smiles (str): SMILES string of the molecule to search in

        Returns:
        list: List of lists containing atom indices for each ring match, or empty list if no matches
        """
        ring_smiles_list = self.ring_dict.get(ring_name, [])
        if not ring_smiles_list:
            print(f"No SMILES patterns found for {ring_name}")
            return []

        # Create molecule from SMILES
        query_mol = Chem.MolFromSmiles(query_mol_smiles)
        if query_mol is None:
            print(f"Could not create molecule from SMILES: {query_mol_smiles}")
            return []

        all_matches = []
        for ring_smi in ring_smiles_list:
            # Create query molecule from ring SMILES
            ring_query = Chem.MolFromSmiles(ring_smi)
            if ring_query is None:
                print(f"Could not create query from SMILES: {ring_smi}")
                continue

            # Find all matches
            matches = query_mol.GetSubstructMatches(ring_query)
            if matches:
                all_matches.extend(list(matches))

        return all_matches

    def check_fg(self, fg, smi):
        fg_smarts_list = self.fg_dict.get(fg, [])
        if not fg_smarts_list:
            print(f"No smarts found for {fg}")
            return False
        # Create the molecule object once
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Could not parse SMILES: {smi}")
            return False
        for fg_smarts in fg_smarts_list:
            patt = Chem.MolFromSmarts(fg_smarts)
            if patt is None:
                continue
            if mol.HasSubstructMatch(patt):
                return True
        return False

    def check_reaction(self, rxn_type, rsmi):
        """
        Checks if a reaction SMILES (rsmi) matches a given reaction type by applying
        reaction templates (SMARTS) and comparing the unmapped products.

        This version includes special handling for single-reactant templates applied to
        multi-reactant inputs (e.g., deprotections where a reagent is present but
        not part of the template's reactant definition).

        Parameters:
        rxn_type (str): The name of the reaction class to check against.
        rsmi (str): The reaction SMILES string (e.g., "reactant1.reactant2>>product1.product2").

        Returns:
        bool: True if the reaction matches the type, False otherwise.
        """
        # 1. Retrieve the list of SMARTS templates for the given reaction type
        rxn_smarts_list = self.reaction_class_dict.get(rxn_type)
        if not rxn_smarts_list:
            print(f"No SMARTS patterns found for reaction type: {rxn_type}")
            return False

        # 2. Parse the input reaction SMILES into reactant and product molecules
        try:
            reactants_smi_str, products_smi_str = rsmi.split(">>")
            reactant_mols = tuple(Chem.MolFromSmiles(s) for s in reactants_smi_str.split('.'))
            
            actual_products_smi_set = set()
            for s in products_smi_str.split('.'):
                mol = Chem.MolFromSmiles(s)
                if mol:
                    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
                    actual_products_smi_set.add(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
            
            if None in reactant_mols or not actual_products_smi_set:
                print(f"Error parsing molecules in reaction SMILES: {rsmi}")
                return False
                
        except ValueError:
            print(f"Invalid reaction SMILES format for '{rsmi}'")
            return False
        except Exception as e:
            print(f"An error occurred while parsing reaction SMILES '{rsmi}': {e}")
            return False

        # 3. Iterate through each SMARTS template for the reaction type
        for rxn_smarts in rxn_smarts_list:
            try:
                template_rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                template_rxn.Initialize()
                num_template_reactants = template_rxn.GetNumReactantTemplates()
            except Exception:
                continue

            # ==================== MODIFIED LOGIC BLOCK START ====================
            #
            # Special case: The template has 1 reactant, but the input reaction has more.
            # This handles cases like deprotections where a reagent is a "spectator"
            # that is not defined in the SMARTS but is present in the reaction SMILES.
            if num_template_reactants == 1 and len(reactant_mols) > 1:
                matching_reactant_indices = []
                
                # Try applying the 1-reactant template to each input reactant individually
                for i, r_mol in enumerate(reactant_mols):
                    predicted_product_sets = template_rxn.RunReactants((r_mol,))
                    
                    # Determine the SMILES of the *other* reactants, which we expect to be unchanged
                    other_reactants_smi = {Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) for j, m in enumerate(reactant_mols) if i != j}

                    for product_set in predicted_product_sets:
                        if not product_set: continue
                        
                        predicted_products_smi = set()
                        for p_mol in product_set:
                            if p_mol:
                                for atom in p_mol.GetAtoms(): atom.SetAtomMapNum(0)
                                predicted_products_smi.add(Chem.MolToSmiles(p_mol, isomericSmiles=True, canonical=True))
                        
                        # The full set of expected products is the union of the predicted transformed
                        # products and the unchanged "other" reactants.
                        hypothetical_products = predicted_products_smi.union(other_reactants_smi)

                        # This must be an exact match to the actual products
                        if hypothetical_products == actual_products_smi_set:
                            matching_reactant_indices.append(i)
                            break # Match found for this reactant, no need to check other outcomes

                # Per your request: The template must apply successfully to *only one* reactant
                # for it to be a valid match.
                if len(matching_reactant_indices) == 1:
                    return True 
                # If it matches 0 or >1 reactants, this template fails. We continue to the next template.

            # ===================== MODIFIED LOGIC BLOCK END =====================
            else:
                # Original logic for when reactant counts match
                # Permute reactants, as RunReactants can be order-sensitive
                for reactants_permutation in itertools.permutations(reactant_mols):
                    try:
                        predicted_product_sets = template_rxn.RunReactants(reactants_permutation)
                    except Exception:
                        continue
                    
                    for product_set in predicted_product_sets:
                        if not product_set: continue
                            
                        predicted_products_smi_set = set()
                        for mol in product_set:
                            if mol:
                                for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
                                predicted_products_smi_set.add(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
            
                        if predicted_products_smi_set and predicted_products_smi_set.issubset(actual_products_smi_set):
                            return True

        return False

    def get_fg_smart(self, fg):
        return self.fg_dict.get(fg, [])

    def get_reaction_smart(self, rxn_type):
        return self.reaction_class_dict.get(rxn_type, [])

    def get_fg_atom_indices(self, fg_name, query_mol_smiles):
        """
        Find the atom indices in the query molecule where a specific functional group matches.

        Parameters:
        fg_name (str): The name of the functional group to search for
        query_mol_smiles (str): SMILES string of the molecule to search in

        Returns:
        list: List of lists containing atom indices for each match, or empty list if no matches
        """
        fg_smarts_list = self.fg_dict.get(fg_name, [])
        if not fg_smarts_list:
            print(f"No SMARTS patterns found for {fg_name}")
            return []

        # Create molecule from SMILES
        query_mol = Chem.MolFromSmiles(query_mol_smiles)
        if query_mol is None:
            print(f"Could not create molecule from SMILES: {query_mol_smiles}")
            return []

        all_matches = []

        for fg_smarts in fg_smarts_list:
            fg_query = Chem.MolFromSmarts(fg_smarts)
            if fg_query is None:
                print(f"Could not create query from SMARTS: {fg_smarts}")
                continue

            # Find all matches
            matches = query_mol.GetSubstructMatches(fg_query)
            if matches:
                all_matches.extend(list(matches))

        return all_matches