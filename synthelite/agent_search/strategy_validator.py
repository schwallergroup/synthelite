import json
import re
from synthelite.agent_search.validator_prompt import validator_prompt
from synthelite.agent_search.base import LLMBase
from synthelite.agent_search.schema import (
    PlanningAttempt,
    RouteFeedback,
    StrategicRoute,
)
from the_retry import retry


class RouteValidator(LLMBase):
    def make_prompt(
        self, route: PlanningAttempt, target_smiles: str, user_constraint: str
    ) -> str:
        retro_reactions = route.retro_reactions
        route_str = "\n".join(f"{i}. {each}" for i, each in enumerate(retro_reactions))
        prompt = (
            validator_prompt.replace("{{TARGET_MOLECULE}}", target_smiles)
            .replace("{{USER_PROMPT}}", user_constraint)
            .replace("{{PROPOSED_SYNTHESIS_PLAN}}", route_str)
        )

        return prompt

    @retry(attempts=3, backoff=5)
    async def validate_route(
        self, route: PlanningAttempt, target_smiles: str, user_constraint: str
    ) -> str:
        prompt = self.make_prompt(route, target_smiles, user_constraint)

        raw_response = await self._query_llm(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        )

        feedback_match = re.search(
            r"<feedback>(.*?)</feedback>", raw_response, re.DOTALL
        )
        if not feedback_match:
            raise ValueError(
                "LLM response does not contain valid <feedback>...</feedback> tags."
            )
        feedback = feedback_match.group(1).strip()
        feedback = json.loads(feedback)
        feedback = RouteFeedback.from_dict(feedback)
        route.feedback = feedback
        return route
