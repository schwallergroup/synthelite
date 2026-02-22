prompt_forward_and_validate = """You are an expert chemist tasked with analyzing and describing a reaction template. Your goal is to interpret a provided SMARTS notation and create a concise, single-sentence description of the reaction.

Here is the reaction template in SMARTS format:

<smarts_reaction>
{{SMARTS_REACTION}}
</smarts_reaction>

which is written in the forward direction, with the reactant fragments are on the left side and the products are on the right side of the reaction arrow '>>'.

Please conduct a thorough analysis of the reaction in <smarts_breakdown> tags. Follow these steps:

1. Break down the SMARTS notation component by component.
2. Separate and provide chemical notation for reactants and products.
3. Explicitly write out the SMARTS notation for reactants and products separately.
4. Identify and number key functional groups or structural elements.
5. Map atoms between reactants and products.
6. Note any stereochemistry changes.
7. Identify leaving groups or new bonds formed.
8. List at least three possible reaction types, including specific name reactions. Consider reactions that might differ only in mechanism or conditions but result in the same transformation. For each, provide pros and cons.
9. Based on the pros and cons, reason about the most likely reaction type(s).
10. Consider both forward and reverse directions of the reaction.
11. Identify potential catalysts or specific conditions implied by the reaction.
12. State any assumptions about reaction conditions or mechanisms.
13. Make specific observations about the SMARTS notation and its implications, avoiding assumptions about specific structures.
14. Describe how you would visualize this reaction if you could draw it.
15. For each major decision or interpretation, explain your thought process.
16. Rate your confidence (Low/Medium/High) for each part of the analysis.

After completing your analysis, evaluate if this reaction template represents a chemically plausible transformation or not. If not, simply output:

<description>
"This reaction template represents a chemically implausible transformation."
</description> 

If it is a valid reaction, formulate a single-sentence description of the reaction in <description> tags. Your description should follow this structure:

<description>
"This reaction involves [general transformation type in forward direction] ([specific name reaction(s)]) of [key reactant(s)] to form [key product(s)], focusing on the [key functional group(s)], and is classified as [reaction type(s)]."
</description>

Ensure that your description:
1. Is written in forward perspective (from reactants to products).
2. Includes all matching reaction types as possibilities if they differ only in mechanism or conditions but not in the outcome of the transformation.
3. Is coherent and clear about the direction of the reaction.
4. Specifies both the general transformation type and the specific name reaction(s). 
5. If there are multiple name reactions which pass by fundamentally different mechanisms also consider those multiple reaction types in your description. 

Here are two examples of a description which you should not copy but use as a reference:

C-O-C(=O)-[n;H0;D3;+0:1](:[#7;a:2]):[c:3]>>[#7;a:2]:[nH;D2;+0:1]:[c:3]
<description>
This reaction involves a protecting group removal (carbamate deprotection) of an N-methoxycarbonyl protected aromatic nitrogen heterocycle to form the corresponding NH-heterocycle, focusing on the N-CO2Me to N-H  transformation, and is classified as a nitrogen deprotection reaction.
</description>

Br-[CH2;D2;+0:1]-[C;H0;D3;+0:2]=O.[NH2;D1;+0:3]-[C;H0;D3;+0:4](=[S;H0;D1;+0:5])-[SH;D1;+0:6]>>[SH;D1;+0:6]-[c;H0;D3;+0:4]1:[n;H0;D2;+0:3]:[c;H0;D3;+0:2]:[cH;D2;+0:1]:[s;H0;D2;+0:5]:1
<description>
This reaction involves a heterocycle formation (modified Hantzsch thiazole synthesis) of α-bromo aldehyde and a thiourea derivative to form a thiol-substituted thiazole, focusing on the formation of the thiazole ring system through C-N and C-S bond formations, and is classified as a nucleophilic substitution-cyclization reaction.
</description>

Begin your response with the reaction analysis, followed by the description. Wrap your thought process in <analysis> tags before providing the final analysis and description."""

prompt_forward = """You are an expert chemist tasked with analyzing and describing a reaction template. Your goal is to interpret a provided SMARTS notation and create a concise, single-sentence description of the reaction.

Here is the reaction template in SMARTS format:

<smarts_reaction>
{{SMARTS_REACTION}}
</smarts_reaction>
which is written in the forward direction, with the reactant fragments are on the left side and the products are on the right side of the reaction arrow '>>'.

Please conduct a thorough analysis of the reaction in <smarts_breakdown> tags. Follow these steps:

1. Break down the SMARTS notation component by component.
2. Separate and provide chemical notation for reactants and products.
3. Explicitly write out the SMARTS notation for reactants and products separately.
4. Identify and number key functional groups or structural elements.
5. Map atoms between reactants and products.
6. Note any stereochemistry changes.
7. Identify leaving groups or new bonds formed.
8. List at least three possible reaction types, including specific name reactions. Consider reactions that might differ only in mechanism or conditions but result in the same transformation. For each, provide pros and cons.
9. Based on the pros and cons, reason about the most likely reaction type(s).
10. Consider both forward and reverse directions of the reaction.
11. Identify potential catalysts or specific conditions implied by the reaction.
12. State any assumptions about reaction conditions or mechanisms.
13. Make specific observations about the SMARTS notation and its implications, avoiding assumptions about specific structures.
14. Describe how you would visualize this reaction if you could draw it.
15. For each major decision or interpretation, explain your thought process.
16. Rate your confidence (Low/Medium/High) for each part of the analysis.

After completing your analysis, formulate a single-sentence description of the reaction in <description> tags. Your description should follow this structure:

"This reaction involves [general transformation type in forward direction] ([specific name reaction(s)]) of [key reactant(s)] to form [key product(s)], focusing on the [key functional group(s)], and is classified as [reaction type(s)]."

Ensure that your description:
1. Is written in forward perspective (from reactants to products).
2. Includes all matching reaction types as possibilities if they differ only in mechanism or conditions but not in the outcome of the transformation.
3. Is coherent and clear about the direction of the reaction.
4. Specifies both the general transformation type and the specific name reaction(s). 
5. If there are multiple name reactions which pass by fundamentally different mechanisms also consider those multiple reaction types in your description. 

Here are two examples of a description which you should not copy but use as a reference:

C-O-C(=O)-[n;H0;D3;+0:1](:[#7;a:2]):[c:3]>>[#7;a:2]:[nH;D2;+0:1]:[c:3]
<description>
This reaction involves a protecting group removal (carbamate deprotection) of an N-methoxycarbonyl protected aromatic nitrogen heterocycle to form the corresponding NH-heterocycle, focusing on the N-CO2Me to N-H  transformation, and is classified as a nitrogen deprotection reaction.
</description>

Br-[CH2;D2;+0:1]-[C;H0;D3;+0:2]=O.[NH2;D1;+0:3]-[C;H0;D3;+0:4](=[S;H0;D1;+0:5])-[SH;D1;+0:6]>>[SH;D1;+0:6]-[c;H0;D3;+0:4]1:[n;H0;D2;+0:3]:[c;H0;D3;+0:2]:[cH;D2;+0:1]:[s;H0;D2;+0:5]:1
<description>
This reaction involves a heterocycle formation (modified Hantzsch thiazole synthesis) of α-bromo aldehyde and a thiourea derivative to form a thiol-substituted thiazole, focusing on the formation of the thiazole ring system through C-N and C-S bond formations, and is classified as a nucleophilic substitution-cyclization reaction.
</description>

Begin your response with the reaction analysis, followed by the description. Wrap your thought process in <analysis> tags before providing the final analysis and description."""

prompt_retro = """You are an expert chemist tasked with analyzing and describing a retrosynthetic reaction template. Your goal is to interpret a provided SMARTS notation and create a concise, single-sentence description of the reaction.

Here is the retrosynthetic reaction template in SMARTS format:

<smarts_reaction>
{{SMARTS_REACTION}}
</smarts_reaction>

Please conduct a thorough analysis of the reaction in <smarts_breakdown> tags. Follow these steps:

1. Break down the SMARTS notation component by component.
2. Separate and provide chemical notation for reactants and products.
3. Explicitly write out the SMARTS notation for reactants and products separately.
4. Identify and number key functional groups or structural elements.
5. Map atoms between reactants and products.
6. Note any stereochemistry changes.
7. Identify leaving groups or new bonds formed.
8. List at least three possible reaction types, including specific name reactions. Consider reactions that might differ only in mechanism or conditions but result in the same transformation. For each, provide pros and cons.
9. Based on the pros and cons, reason about the most likely reaction type(s).
10. Consider both forward and reverse directions of the reaction.
11. Identify potential catalysts or specific conditions implied by the reaction.
12. State any assumptions about reaction conditions or mechanisms.
13. Make specific observations about the SMARTS notation and its implications, avoiding assumptions about specific structures.
14. Describe how you would visualize this reaction if you could draw it.
15. For each major decision or interpretation, explain your thought process.
16. Rate your confidence (Low/Medium/High) for each part of the analysis.

After completing your analysis, formulate a single-sentence description of the reaction in <description> tags. Your description should follow this structure:

"This retrosynthetic disconnection involves [general transformation type] ([specific name reaction(s)]) of [key product(s)] to form [key reactant(s)], focusing on the [key functional group(s)], and is classified as [reaction type(s)]."

Ensure that your description:
1. Maintains a retrosynthetic perspective (from products to reactants).
2. Includes all matching reaction types as possibilities if they differ only in mechanism or conditions but not in the outcome of the transformation.
3. Is coherent and clear about the direction of the reaction.
4. Specifies both the general transformation type and the specific name reaction(s). 
5. If there are multiple name reactions which pass by fundamentally different mechanisms also consider those multiple reaction types in your description. 

Here are two examples of a description which you should not copy but use as a reference:

<description>
This retrosynthetic disconnection involves oxidative coupling (Glaser coupling, Eglinton coupling, or Hay coupling) of a conjugated diyne to form two terminal alkyne reactants, focusing on the alkyne functional groups, and is classified as an oxidative alkyne dimerization reaction.
</description>

<description>
This retrosynthetic disconnection involves a multicomponent heterocycle fragmentation (modified Hantzsch-type reaction) of a 6-membered sulfur-nitrogen containing heterocycle to form a ketone, β-keto sulfone, aldehyde, and ammonia, focusing on the C-C and C-N bond formations that create the ring system, and is classified as a multicomponent condensation/cyclization reaction.
</description>

Begin your response with the reaction analysis, followed by the description. Wrap your thought process in <analysis> tags before providing the final analysis and description."""
