"""
{
   'route_id': 1,
   'steps': [
      {
         'step_number': 1,
         'step_description': <disconnection description in natural languages>,
         'step_reaction': <retro reaction SMILES>,
      },
      {
         'step_number': 2,
         'step_description': <disconnection description in natural languages>,
         'step_reaction': <retro reaction SMILES>,
      },
      ...
   ]
}
"""

ranking_prompt = """You are an experienced organic chemist tasked with evaluating a retrosynthesis strategy for a target molecule. You will be given a list of in-development retrosynthetic routes and their intended strategy. Your goal is to assess how well each route aligns with the user's requirement and assign a score based on the alignment and feasibility of the transformations. You assessment will be used to choose the most promising routes for further development. 

First, review the context of the synthesis, including the target molecule, constraints for the synthesis, and potentially some additional information about the synthesis strategy:

<synthesis_context>
{{SYNTHESIS_STRATEGY}}
</synthesis_context>

The partially constructed candidate routes are represented as a list of dictionaries as followed. Note that not just one but several routes can be provided, and each route candidate is represented by a list of steps, where each step contains a description and a reaction SMILES in retro direction:

<partially_constructed_candidate_routes>
{{CANDIDATE_ROUTES}}
</partially_constructed_candidate_routes>
Note that if the reaction SMILES of a step is missing but the description is present, it means that the step is still in-planning but the exact reaction is not yet pinpointed.

Your task is to analyze the candidate routes and assign a score to each route. For each route, evaluate the chemical feasibility of each step's reaction SMILES ( if available in 'step_reaction') and its alignment with the corresponding step description, and evaluate if the overall route aligns with the synthesis context. Consider the following points in your evaluation:

For each route, follow those steps:

1. Analyze the retro reaction SMILES (if available) of each step in the route:
   - Identify the specific bonds that have been broken or formed.
   - Note any functional group changes or additions.
   - Describe the fragments that are resulted from the transformation and evaluate the feasibility of the subsequent steps based on the fragments.
   Based on this analysis, determine if the reaction SMILES is chemically feasible and aligns with the corresponding description provided in 'step_description'. Assign a score between 0 and 10 for each step, where 10 indicates perfect alignment and chemical feasibility, and 0 indicates no alignment or not chemically feasible transformation.
   
2. Based on the analysis of the reaction SMILES and the descriptions of all steps, provide an overall evaluation of the route:
   - The feasibility of the route as a whole, taking into account also the future steps whose reaction SMILES are not yet available but the descriptions are provided.
   - How closely the route follows the user's requirements.
   - If protecting groups are introduced, are they necessary?
   - How likely the route would lead to accessible starting materials.
   - The estimated cost of the synthesis, taking into account the future steps.
   - Identify any potential issues or improvements that could be made.
   Based on this analysis, assign a score between 0 and 10 for each route.

After analyzing all routes, provide a final ranking of the routes based on their scores, from the most promising to the least promising. Present your final evaluation in the following format:

<route_evaluation>
[
   {
      'route_id': 1,
      'steps': [
         {
            'step_number': 1,
            'step_description': <disconnection description in natural languages>,
            'step_reaction': <retro reaction SMILES>,
            'step_evaluation': <summary of the evaluation of the step, including chemical feasibility and alignment with the description>,
            'step_score': <score between 0 and 10>,
         },
         {
            'step_number': 2,
            'step_description': <disconnection description in natural languages>,
            'step_reaction': <retro reaction SMILES>,
            'step_evaluation': <summary of the evaluation of the step, including chemical feasibility and alignment with the description>,
            'step_score': <score between 0 and 10>,
         },
         ...
      ],
      'route_evaluation': <summary of the evaluation of the route>,
      'route_score': <score between 0 and 10>,
   },
   {
      'route_id': 2,
      ...
   }
]
</route_evaluation>

<justification>
[Your detailed justification for route candidate 1]
[Your detailed justification for route candidate 2]
...
</justification>

The final ranking should be given as a python list of 'route_id' in integer, ordered from the most promising to the least promising:

<route_ranking>
[<list of route_id in integer, ordered from the most promising to the least promising>]
</route_ranking>

You MUST provide a route_ranking within the <route_ranking> tag.
Remember to base your evaluation on sound chemical reasoning and the planned synthesis strategy. Be thorough in your analysis, as this will inform crucial decisions in the retrosynthesis process."""

scoring_prompt = """You are an experienced organic chemist tasked with evaluating a retrosynthesis strategy for a target molecule. Your goal is to assess how well the current disconnection aligns with the given synthesis strategy.

First, review the synthesis strategy developed based on user constraints including the target molecule, constraints for the synthesis, key transformations and an estimated number of steps:

<synthesis_strategy>
{{SYNTHESIS_STRATEGY}}
</synthesis_strategy>
Note: This strategy serves as an overall guiding framework rather than a detailed step-by-step plan.

The partially constructed route, including the current last step, is represented by the following reaction SMILES:

<partially_constructed_route>
{{PREVIOUS_SMILES}}
</partially_constructed_route>
Noted that the reactions are written in the retro direction, with products on the left and reactants on the right of the reaction arrow.


Your task is to analyze the current disconnection by examining the partially constructed route, with particular attention to the most recent transformation. Evaluate how this aligns with the overall synthesis strategy, considering the current depth and its relation to the steps outlined in the strategy.

Follow those steps:

1. Compare the current intermediate (the last molecule(s) in the partially constructed route) to its immediate precursor:
   - Identify the specific bonds that have been broken or formed.
   - Note any functional group changes or additions.

2. Assess how closely the current disconnection follows the planned strategy:
   - Compare this step to the synthesis strategy and determine if it matches the relatively corresponding step in the strategy.
   - Identify any deviations from the planned route. Note that the provided strategy serves as an overall guiding framework rather than a strict step-by-step plan, so deviations might still be acceptable if the route still aligns with the overall synthesis goals.

4. Evaluate if the disconnection is appropriate at the current stage of the retrosynthesis, compared with the strategy provided.

5. Determine if the transformation makes sense from a chemical standpoint.

6. Identify any potential issues or improvements that could be made:
   - Consider how changes might affect subsequent steps.

After your analysis, provide a short justification for your assessment and assign a numerical score between 0 and 10, where 10 indicates chemical feasibility and perfect alignment with the strategy and 0 indicates no alignment at all or not chemically feasible transformation. Consider each point in your analysis when assigning the final score.

Present your final evaluation in the following format:

<justification>
[Your detailed justification here]
</justification>

<score>
[Your numerical score between 0 and 10]
</score>

Remember to base your evaluation on sound chemical reasoning and the planned synthesis strategy. Be thorough in your analysis, as this will inform crucial decisions in the retrosynthesis process."""

scoring_prompt_old = """You are an experienced organic chemist tasked with evaluating a retrosynthesis strategy for a target molecule. Your goal is to assess how well the current disconnection aligns with the given synthesis strategy.

First, review the synthesis strategy developed based on user constraints including the target molecule, constraints for the synthesis, key transformations and an estimated number of steps:

<synthesis_strategy>
{{SYNTHESIS_STRATEGY}}
</synthesis_strategy>

The partially constructed route, including the current last step, is represented by the following reaction SMILES:

<partially_constructed_route>
{{PREVIOUS_SMILES}}
</partially_constructed_route>

Now, consider the current state of the retrosynthesis:

<current_depth>
{{CURRENT_DEPTH}}
</current_depth>

Your task is to analyze the current disconnection by examining the partially constructed route, with particular attention to the most recent transformation. Evaluate how this aligns with the overall synthesis strategy, considering the current depth and its relation to the steps outlined in the strategy.

Follow those steps:

1. Compare the current intermediate (the last molecule(s) in the partially constructed route) to its immediate precursor:
   - Identify the specific bonds that have been broken or formed.
   - Note any functional group changes or additions.

2. Assess how closely the current disconnection follows the planned strategy:
   - Compare this step to the synthesis strategy and determine if it matches the expected transformations.
   - Identify any deviations from the planned route.

4. Evaluate if the disconnection is appropriate for the current depth in the retrosynthesis.

5. Determine if the transformation makes sense from a chemical standpoint.

6. Identify any potential issues or improvements that could be made:
   - Consider how changes might affect subsequent steps.

After your analysis, provide a short justification for your assessment and assign a numerical score between 0 and 10, where 10 indicates perfect alignment with the strategy and 0 indicates no alignment at all. Consider each point in your analysis when assigning the final score.

Present your final evaluation in the following format:

<justification>
[Your detailed justification here]
</justification>

<score>
[Your numerical score between 0 and 10]
</score>

Remember to base your evaluation on sound chemical reasoning and the planned synthesis strategy. Be thorough in your analysis, as this will inform crucial decisions in the retrosynthesis process."""
