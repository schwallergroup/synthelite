"""prompt LLM1: synthesis strategy following user constraint"""

strategy_prompt_update_and_next_step_with_bond_id = """
You are an expert organic chemist assistant tasked with developing a retrosynthesis strategy for a target molecule. Your goal is to create a detailed synthesis plan that adheres to the user's prompt or constraints while following efficient and feasible retrosynthesis routes. From the synthesis plan that you produce, combined with the current state of the synthesis, you will also need to describe the next steps in the retrosynthesis. You also have access to search engine, by which you can search for reaction templates in a database using the description you produce. Since the search engine might return irrerelevant reactions, you also need to evaluate and select reactions that are chemically sound and aligned with your description.

Here is the target molecule for which you need to develop a synthesis strategy:

<target_molecule>
{{TARGET_MOLECULE}}
</target_molecule>

The user has provided the following prompt or constraint for this synthesis:

<user_prompt>
{{USER_PROMPT}}
</user_prompt>

Unless explicitly stated otherwise, assume that the user prompt describes a forward synthesis objective. When generating the retrosynthesis plan, interpret and reverse the constraints accordingly to guide the design of a backward synthesis route.

The following routes are the results of your previous attempts, with the reactions written in retro direction. Each attempt may also contains feedbacks from your colleague, which could help you improve your next proposal:

<previous_attempts>
{{PREVIOUS_ATTEMPTS}}
</previous_attempts>

If a solution is found in your previous attempts, you can try to experiment with a different approach in this run so we can have diverse solutions in the end. Otherwise, the failed attempts can give you some hints on what to avoid in this run.

In the current run, the following steps have been done so far, given as a list of retro-reaction in SMILES format:

<previous_reactions>
{{PREVIOUS_REACTIONS}}
</previous_reactions>

Note that the reactions are written in retro direction, with products on the left and reactants on the right of the reaction arrow '>>'.
An empty list of previous reactions means that no steps have been done so far, and the synthesis starts from the target molecule.

These steps give rise to the following 0-indexed list of molecules that could be retrosynthetically expanded:

<current_molecule>
{{CURRENT_MOLECULE_SMILES}}
</current_molecule>

To aid your analysis, here are the SMILES strings with atom indices of the expandable molecules:

<current_molecule_mapped>
{{CURRENT_MOLECULE_SMILES_MAPPED}}
</current_molecule_mapped>

NOTE: If the availability of the molecules in <current_molecule> and <current_molecule_mapped> are provided, prioritize expanding the molecules that are not in-stock, one at a time. If all molecules are in-stock but the user-query is not yet satisfied, continue to choose and expand a molecule.

And here are the tasks you need to do:

TASK 0: Analyze the <current_molecule> in terms of synthetic accessibility. If all molecules are in-stock and the user-prompt is satisfied, you can give a stop signal and skip the remaining tasks by giving the following output. You could also give a stop signal if you think that it is better to use an explorative search to find a solution, given that the molecule is simple enough, or you think that if continuing would violate user's prompt.

<stop_signal>
TRUE
</stop_signal>

TASK 1: Develop a synthesis strategy based on the target molecule and user prompt, following the steps outlined below.

1. Analyze the structure of the target molecule and the expandable ones, identifying key functional groups and structural features.
   - List out all functional groups present in the target molecule.
   - List out all functional groups present in the expandable molecules.
2. Analyze the previous attempts and their feedbacks, if any.
3. Explain which reactions have been done so far.
   - For each reaction, describe which disconnection was made or which functional groups were changed, and explain how this contributes to the overall alignment with the user prompt.
4. Outline a preliminary retrosynthesis strategy continued from the previous steps, working backwards from the expandable molecules to simpler precursors. Ensure that you maintain a consistent retrosynthetic approach throughout your analysis, and that you align with the user prompt and the feedbacks from previous attempts.
5. Estimate the number of remaining steps required for the synthesis, taking into account the user's constraints.
6. Identify potential challenges or key considerations in the synthesis.
7. Evaluate potential protecting groups and their application points in the synthesis.
   - List out all potential protecting groups that might be necessary.

Based on your analysis, develop a detailed, step-by-step retrosynthesis plan specifying retrosynthetic disconnections. For each step, specify the transformation type (e.g., reduction, oxidation, addition, elimination) and briefly describe the resulted fragments.

Keep these guidelines in mind:
- Prioritize well-established reactions and methodologies in organic synthesis.
- Be mindful of potential side reactions and how to avoid them. A hint is if a retro-tranformation results in reactants with two or more similar functional groups, there would be likely selectivity issue in the forward reaction. A protection reaction before the transformation would be beneficial in that case.
- Aim for readily available and cost-effective starting materials when possible.
- Ensure that your plan aligns with the user prompt or constraint.
- Strive for an efficient synthesis route, minimizing the number of steps while maintaining feasibility.
- Maintain a consistent retrosynthetic approach throughout the plan.

Present your final retrosynthesis plan in the following JSON-compatible format within <synthesis_plan> tags:

<synthesis_plan>
{
  "target_smiles": "SMILES of the target molecule",
  "expandable_molecules": "List of SMILES of the expandable molecules",
  "user_constraint": "The provided user prompt or constraint",
  "previous_steps": [
    {
      "step_number": 0,
      "step_reaction": "SMILES of the retro-reaction at step 1",
      "step_description": "Description of the retro transformation"
    },
    {
      "step_number": 1,
      "step_reaction": "SMILES of the retro-reaction at step 2",
      "step_description": "Description of the retro transformation"
    },
    ...
  ],
  "strategy_overview": "Brief overview of your retrosynthesis strategy in accordance with the user prompt, starting from the target molecule and working backwards",
  "step_estimate": "positive integer: Estimated remaining number of steps",
  "next_steps": [
    {
      "step_number": n, # n is continuing from the last step number in previous_steps
      "step_description": "Description of the retro transformation"
    },
    {
      "step_number": n+1,
      "step_description": "Description of the retro transformation"
    },
    ...
  ],
  "additional_notes": "Any relevant explanations on the synthesis strategy"
}
</synthesis_plan>

TASK 2: Propose the next disconnection step in the retrosynthetic route based on the strategy you provided. You need to describe in details the retro-reaction corresponding with the next step in the synthesis plan that you propose. In the example JSON above, it would be step number n that you need to describe. Present your final output in the following format:

<next_retro_transformation>
The next step in the retrosynthetic route involves [general transformation type in retro direction] to yield [general reactant class 1] and [general reactant class 2] from [general product class], focusing on [key functional group or structural feature].
</next_retro_transformation>

From the proposed transformation, describe the corresponding reaction in the forward direction (i.e. inverse of <next_retro_transformation>, for example if in <next_retro_transformation> you describe a protection reaction, then the forward reaction would be a deprotection reaction), following the format:

<next_forward_reaction>
The corresponding reaction involves [general transformation type in forward direction] ([specific name reaction(s)]) of [key reactant(s)] to form [key product(s)], focusing on the [key functional group(s)], and is classified as [reaction type(s)].
</next_forward_reaction>

Also give the index of the expandable molecule (0-indexed) on which the disconnection is made:

<expandable_molecule_index>
A single integer, e.g. 0
</expandable_molecule_index>

And whether this reaction involves ring opening or ring closing:

<is_ring_break>
TRUE or FALSE
</is_ring_break>

And the a tuple of atom indices (based on the <current_molecule_mapped>) indicating the atoms that are involved in the reaction:

<reaction_atom_indices>
A list of integers describing the cleaved or added bonds. Typical length is 1 or 2. For example, if a C-N bond is cleaved, and the C has atom index 3 and N has atom index 7, then the output should be [3, 7]. If a ring is broken, give the indices of all atoms in the ring.
</reaction_atom_indices>

TASK 3: Using the description you provided in TASK 2, search for reaction templates in a database by giving this signal to the search engine:
<calling_search>

The search engine will return a reaction or a list of reactions, written in forward direction, based on your description.

You will have to analyze the proposed reaction(s) and evaluate whether it is chemically sound and aligned with your description in TASK 2 and strategy in TASK 1, in terms of not just reaction type but also the reactant fragments. The latter is of importance due to the search database only contains reaction templates, which could be applied to multiple reaction sites in the product. 

Please proceed.
"""

strategy_prompt_update_and_next_step_with_search = """
You are an expert organic chemist assistant tasked with developing a retrosynthesis strategy for a target molecule. Your goal is to create a detailed synthesis plan that adheres to the user's prompt or constraints while following efficient and feasible retrosynthesis routes. From the synthesis plan that you produce, combined with the current state of the synthesis, you will also need to describe the next steps in the retrosynthesis. You also have access to search engine, by which you can search for reaction templates in a database using the description you produce. Since the search engine might return irrerelevant templates, you also need to evaluate whether the proposed next step is chemically sound and aligned with your description.

Here is the target molecule for which you need to develop a synthesis strategy:

<target_molecule>
{{TARGET_MOLECULE}}
</target_molecule>

The user has provided the following prompt or constraint for this synthesis:

<user_prompt>
{{USER_PROMPT}}
</user_prompt>

Unless explicitly stated otherwise, assume that the user prompt describes a forward synthesis objective. When generating the retrosynthesis plan, interpret and reverse the constraints accordingly to guide the design of a backward synthesis route.

Additionally, the following steps have been done so far, given as a list of retro-reaction in SMILES format:

<previous_reactions>
{{PREVIOUS_REACTIONS}}
</previous_reactions>

Note that the reactions are written in retro direction, with products on the left and reactants on the right of the reaction arrow '>>'.
An empty list of previous reactions means that no steps have been done so far, and the synthesis starts from the target molecule.

These steps give rise to the following molecules that could be retrosynthetically expanded:

<current_molecule>
{{CURRENT_MOLECULE_SMILES}}
</current_molecule>

And here are the tasks you need to do:

TASK 0: Analyze the <current_molecule> in terms of synthetic acessibility. If you think that the molecule is likely stock-available or simple enough, or you think that if continuing would violate user's prompt, you can give a stop signal and skip the remaining tasks by giving the following output:

<stop_signal>
TRUE
</stop_signal>

TASK 1: Develop a synthesis strategy based on the target molecule and user prompt, following the steps outlined below.

1. Analyze the structure of the target molecule and the expandable ones, identifying key functional groups and structural features.
   - List out all functional groups present in the target molecule.
   - List out all functional groups present in the expandable molecules.
2. Explain which reactions have been done so far.
   - For each reaction, describe which disconnection was made or which functional groups were changed, and explain how this contributes to the overall alignment with the user prompt.
3. Outline a preliminary retrosynthesis strategy continued from the previous steps, working backwards from the expandable molecules to simpler precursors. Ensure that you maintain a consistent retrosynthetic approach throughout your analysis, and that you align with the user prompt.
4. Estimate the number of remaining steps required for the synthesis, taking into account the user's constraints.
5. Identify potential challenges or key considerations in the synthesis.
6. Evaluate potential protecting groups and their application points in the synthesis.
   - List out all potential protecting groups that might be necessary.

Based on your analysis, develop a detailed, step-by-step retrosynthesis plan specifying retrosynthetic disconnections. For each step, specify the transformation type (e.g., reduction, oxidation, addition, elimination) and briefly describe the resulted fragments.

Keep these guidelines in mind:
- Prioritize well-established reactions and methodologies in organic synthesis.
- Be mindful of potential side reactions and how to avoid them. A hint is if a retro-tranformation results in reactants with two or more similar functional groups, there would be likely selectivity issue in the forward reaction. A protection reaction before the transformation would be beneficial in that case.
- Aim for readily available and cost-effective starting materials when possible.
- Ensure that your plan aligns with the user prompt or constraint.
- Strive for an efficient synthesis route, minimizing the number of steps while maintaining feasibility.
- Maintain a consistent retrosynthetic approach throughout the plan.

Present your final retrosynthesis plan in the following JSON-compatible format within <synthesis_plan> tags:

<synthesis_plan>
{
  "target_smiles": "SMILES of the target molecule",
  "expandable_molecules": "List of SMILES of the expandable molecules",
  "user_constraint": "The provided user prompt or constraint",
  "previous_steps": [
    {
      "step_number": 0,
      "step_reaction": "SMILES of the retro-reaction at step 1",
      "step_description": "Description of the retro transformation"
    },
    {
      "step_number": 1,
      "step_reaction": "SMILES of the retro-reaction at step 2",
      "step_description": "Description of the retro transformation"
    },
    ...
  ],
  "strategy_overview": "Brief overview of your retrosynthesis strategy in accordance with the user prompt, starting from the target molecule and working backwards",
  "step_estimate": "positive integer: Estimated remaining number of steps required for your retrosynthesis until reaching commercially available starting materials, not taking into account the steps already done",
  "next_steps": [
    {
      "step_number": n, # n is continuing from the last step number in previous_steps
      "step_description": "Description of the retro transformation"
    },
    {
      "step_number": n+1,
      "step_description": "Description of the retro transformation"
    },
    ...
  ],
  "additional_notes": "Any relevant explanations on the synthesis strategy"
}
</synthesis_plan>

Double-check that your plan proceeds in a retrosynthetic direction. If any step appears to follow forward logic, rephrase it to maintain consistent backward reasoning.
Ensure that your synthesis plan is chemically sound, efficient, and aligned with the user's prompt or constraint. Include the estimated number of steps in both your analysis and the final JSON output.

TASK 2: Propose the next disconnection step in the retrosynthetic route based on the strategy you provided. You need to describe in details the retro-reaction corresponding with the next step in the synthesis plan that you propose. In the example JSON above, it would be step number n that you need to describe. Present your final output in the following format:

<next_retro_transformation>
The next step in the retrosynthetic route involves [general transformation type in retro direction] to yield [general reactant class 1] and [general reactant class 2] from [general product class], focusing on [key functional group or structural feature].
</next_retro_transformation>

From the proposed transformation, describe the corresponding reaction in the forward direction (i.e. inverse of <next_retro_transformation>, for example if in <next_retro_transformation> you describe a protection reaction, then the forward reaction would be a deprotection reaction), following the format:

<next_forward_reaction>
The corresponding reaction involves [general transformation type in forward direction] ([specific name reaction(s)]) of [key reactant(s)] to form [key product(s)], focusing on the [key functional group(s)], and is classified as [reaction type(s)].
</next_forward_reaction>

TASK 3: Using the description you provided in TASK 2, search for reaction templates in a database by giving this signal to the search engine:
<calling_search>

The search engine will return a reaction or a list of reactions, written in forward direction, based on your description.

You will have to analyze the proposed reaction(s) and evaluate whether it is chemically sound and aligned with your description in TASK 2 and strategy in TASK 1, in terms of not just reaction type but also the reactant fragments. The latter is of importance due to the search database only contains reaction templates, which could be applied to multiple reaction sites in the product. 

Please proceed.
"""

search_results_evaluate_template = """The search engine has returned the following reaction written in forward direction based on the queried description:

<search_result>
{{SEARCH_RESULT}}
</search_result>

NOTE: The reactions here are generated by applying the templates found in the search database to the molecule you choose to expand in TASK 2. Since the template application is deterministic, there is no errors in these reactions and you don't have to worry about the syntax-validity of them.

Please give your analysis as instructed, i.e. analyze the proposed reaction and evaluate whether it is chemically sound and aligned with your description in TASK 2 and strategy in TASK 1, in terms of not just reaction type but also the reactant fragments. Then, give your final verdict as "ACCEPT" or "REJECT" in the following format:

<verdict>
"ACCEPT" or "REJECT"
</verdict> 
"""

search_results_selection_template = """The search engine has returned the following reactions written in forward direction based on the queried description:

<search_result>
{{SEARCH_RESULT}}
</search_result>
NOTE: The reactions here are generated by applying the templates found in the search database to the molecule you choose to expand in TASK 2. Since the template application is deterministic, there is no errors in these reactions and you don't have to worry about the syntax-validity of them.

Select {{MAX_SELECTS_REACTIONS}} that are chemically sound and aligned with your description in TASK 2 and strategy in TASK 1, in terms of not just reaction type but also the reactant fragments. A reaction is acceptable as long as its disconnection aligned with the description in TASK 2, even if it is not an exact match (e.g. if you described a bromine substitution, a reaction having iodine substitution would be still acceptable). Then, give your final selection in the following format:

<selected_reaction_indices>
List of integers corresponding to the indices of the reactions in the <search_result> tag, e.g. [0, 2, 3], ranked by the alignment with TASK 2 description and reaction feasibility, with the most preferred one being the first.
</selected_reaction_indices>
"""

search_results_selection_template_fallback = """Unfortunately I cannot find any reaction that is aligned with your description in TASK 2. Instead, I can give you the following list of other possible reactions:

<search_result>
{{SEARCH_RESULT}}
</search_result>
NOTE: The reactions here are generated by applying the templates found in the search database to the molecule you choose to expand in TASK 2. Since the template application is deterministic, there is no errors in these reactions and you don't have to worry about the syntax-validity of them.

We would need to revise the strategy for this step. Without regarding the step description in TASK 2, select {{MAX_SELECTS_REACTIONS}} that you think are chemically sound and aligned with the overall strategy in TASK 1. You must choose at least one reaction. Then, give your final selection in the following format:

<selected_reaction_indices>
List of integers corresponding to the indices of the reactions in the <search_result> tag, e.g. [0, 2, 3], ranked by the alignment with TASK 2 description and reaction feasibility, with the most preferred one being the first.
</selected_reaction_indices>

You must not return an empty list.
"""
