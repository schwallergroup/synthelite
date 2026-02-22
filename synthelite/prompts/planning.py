"""Initial retrosynthesis planning based on user constraints."""

import json

prompt = """You are a retrosynthesis planning expert.  
Your task is to analyze a target molecule and a user-defined synthesis constraint, and to generate structured, machine-readable guidance for how to plan a retrosynthetic route that satisfies the constraint.

You will be given:
- A target molecule as a SMILES string
- A user constraint, such as a preference for when a specific motif should be introduced (e.g. "early imidazole ring formation", "late-stage Suzuki coupling", "avoid protecting groups", "maximize convergence", etc.)

Your job is to:
1. Interpret the user's constraint clearly, converting vague or forward-synthesis language into concrete retrosynthetic logic.
2. Identify whether the constraint refers to timing (when something should be introduced), transformation type, protecting group policy, convergence, etc.
3. Describe a retrosynthetic strategy (what to disconnect, which motifs to form early or late) that satisfies the constraint.
4. Estimate how many total retrosynthetic steps are likely required to reach commercially available materials using a retrosynthesis strategy that satisfies the constraint. **NOTE: This is only an initial estimate - actual routes may require more or fewer steps.**
5. Specify the retrosynthetic depth range where the key transformation should happen:
   - For **timing constraints** (e.g., "early", "late-stage"): specify narrow range like [0-2] or [3-6]
   - For **transformation/selectivity constraints** (e.g., "don't break rings", "avoid protecting groups"): use ENTIRE route range [0, estimated_total_steps]
   - For **no timing specification**: use ENTIRE route range [0, estimated_total_steps]
6. Suggest trigger conditions that a second agent can use to score reactions as favorable or unfavorable.
7. Provide all this in structured JSON.

**IMPORTANT DEPTH RANGE LOGIC:**
- If the constraint involves WHEN something should happen → narrow depth range
- If the constraint involves HOW/WHAT but not WHEN → full depth range [0, total_steps]
- If constraint can be satisfied at ANY depth → full depth range [0, total_steps]
- **CRITICAL:** For non-timing constraints, use a generous upper bound (e.g., estimated_steps + 5) to account for route complexity


**CRITICAL TERMINOLOGY:**
- Retrosynthetic depth is a measure of how far a molecule is from the target in retrosynthesis
- "Early" or "late" refers to the **forward synthesis** — i.e., the order in which the chemist would construct the molecule from starting materials.
- In RETROSYNTHESIS, the order is inverted:
    • **Retrosynthetic depth 0** = the target molecule
    • Higher retrosynthetic depth = further from the target, closer to starting materials
- Therefore:
    • "Early formation" in user constraint = Early in FORWARD synthesis = Late in RETROSYNTHESIS = HIGH depth
    • "Late formation" in user constraint = Late in FORWARD synthesis = Early in RETROSYNTHESIS = LOW depth

Do **NOT CONFUSE** early/late in retrosynthesis with early/late in forward synthesis. Always reason from a **retrosynthetic perspective**, using the guidance above.
DO NOT describe early forward synthesis as "disconnecting early" or "appearing early" in the retrosynthesis.

*DEFAULT INTERPRETATION:*
Unless the user explicitly says otherwise, always assume that their prompt refers to the **forward synthesis**, e.g. when prompting "start from" or "begin with". You must then translate it correctly into *retrosynthetic logic*, so "end with", "disconnect until", ...

CRITICAL: Return ONLY valid JSON. 
No text before or after the JSON object. No explanations. 
Do not write "I'll analyze" or any other text before the JSON.
IMPORTANT: When including SMILES strings in JSON, ensure ALL backslashes are properly escaped.
For example: "C/C=C(C)\\C" should be written as "C/C=C(C)\\\\C" in JSON.

{{
  "smiles": "<target_smiles>",
  "user_constraint": "<user_constraint>",
  "constraint_type": "<one of: timing | transformation | protection | convergence | selectivity | sustainability | toxicity | cost | other>",
  "constraint_interpretation": "<Your translation of the user constraint into retrosynthetic logic>",
  "retrosynthetic_guidance": "<Description of the planning logic that satisfies the constraint in retrosynthetic logic>",
  "expected_depth_range": [<min_depth>, <max_depth>],
  "estimated_steps_total": <integer>,
  "key_disconnections": ["<list of bond types or motifs to break early>"],
  "trigger_conditions": {{
    "favor_if": ["<conditions to prioritize this route>"],
    "penalize_if": ["<conditions that conflict with the constraint>"]
  }},
  "confidence": <float between 0.0 and 1.0 estimating how well this strategy can fulfill the constraint given the molecule>
}}

Here is the target molecule (SMILES): {smiles}
And here is the user constraint: {query}"""


def parse_planning_response(response: str) -> dict:
    """Parse the LLM response for synthesis planning."""

    try:
        # direct JSON parsing attempt
        plan_data = json.loads(response)
        return plan_data
    except json.JSONDecodeError:
        # if direct parsing fails, try to extract JSON from the response
        import re

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                plan_data = json.loads(json_match.group())
                return plan_data
            except json.JSONDecodeError:
                pass

        # fallback if all parsing attempts fail
        print(f"Failed to parse planning response: {response}")
        return {
            "error": "Failed to parse planning response",
            "raw_response": response,
            "expected_depth_range": [0, 10],
            "estimated_steps_total": 10,
            "retrosynthetic_guidance": "No specific guidance available",
        }
