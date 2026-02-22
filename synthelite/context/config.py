""" Module containing a class for encapsulating the settings of the tree search
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

from synthelite.agent_search.strategy_validator import RouteValidator
from synthelite.context.policy import ExpansionPolicy, FilterPolicy
from synthelite.context.scoring import ScorerCollection
from synthelite.context.stock import Stock
from synthelite.query_embeddings.embeddings_search import FastTemplateSearchEngine
from synthelite.utils.clogging import logger

if TYPE_CHECKING:
    from synthelite.utils.type_utils import Any, Dict, List, Optional, StrDict, Union


@dataclass
class _PostprocessingConfiguration:
    min_routes: int = 5
    max_routes: int = 25
    all_routes: bool = False
    route_distance_model: Optional[str] = None
    route_scorers: List[str] = field(default_factory=lambda: [])
    scorer_weights: Optional[List[float]] = field(default_factory=lambda: None)


@dataclass
class _SteerConfiguration:
    model: str = "gpt-4o"
    llm_weight: float = 0
    llm_prob: float = 0
    llm_prompt: Optional[str] = None
    llm_vision: bool = False
    wandb_project: Optional[str] = None


@dataclass
class _LLMGuidanceConfiguration:
    n_runs: int = 1
    enabled: bool = False
    step_limit: int = 25  # LLM-guided step limit in Phase 1
    strategy_planner: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.1,
            "backend": "openrouter",
        }
    )
    template_search: Dict[str, Any] = field(
        default_factory=lambda: {
            "csv_path": "",
            "similarity_threshold": 0.3,
            "max_templates": 10,
        }
    )
    next_step_generator: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.3,
            "backend": "openrouter",
        }
    )
    strategic_scorer: Dict[str, Any] = field(
        default_factory=lambda: {
            "model": "claude-3-5-sonnet-20241022",
            "probability": 0.2,
            "weight": 1.0,
            "score_function": "dynamic_strategy_heuristic_rank",
            "score_args": {
                "repetitions": 2,
            },
        }
    )
    commercial_availability: Dict[str, Any] = field(
        default_factory=lambda: {"check_depth": 3, "early_termination": True}
    )


@dataclass
class _SearchConfiguration:
    algorithm: str = "mcts"
    algorithm_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "W": 1,
            "C": 1.4,
            "default_prior": 0.5,
            "use_prior": True,
            "prune_cycles_in_search": True,
            "search_rewards": ["state score"],
            "immediate_instantiation": (),
            "mcts_grouping": None,
            "search_rewards_weights": [],
        }
    )
    max_transforms: int = 6
    iteration_limit: int = 100
    # neural_guided_iteration_limit: int = 20
    query_explore_iteration_limit: int = 50
    time_limit: int = 1200
    return_first: bool = False
    exclude_target_from_stock: bool = True
    break_bonds: List[List[int]] = field(default_factory=list)
    freeze_bonds: List[List[int]] = field(default_factory=list)
    break_bonds_operator: str = "and"
    steer: _SteerConfiguration = _SteerConfiguration()
    llm_guidance: _LLMGuidanceConfiguration = _LLMGuidanceConfiguration()


@dataclass
class Configuration:
    """
    Encapsulating the settings of the tree search, including the policy,
    the stock, the loaded scorers and various parameters.
    """

    search: _SearchConfiguration = field(default_factory=_SearchConfiguration)
    post_processing: _PostprocessingConfiguration = field(
        default_factory=_PostprocessingConfiguration
    )
    stock: Stock = field(init=False)
    expansion_policy: ExpansionPolicy = field(init=False)
    filter_policy: FilterPolicy = field(init=False)
    scorers: ScorerCollection = field(init=False)
    template_search: Optional[FastTemplateSearchEngine] = None
    route_validator: Optional[
        RouteValidator
    ] = None  # Placeholder for future route validator

    def __post_init__(self) -> None:
        self.stock = Stock()
        self.expansion_policy = ExpansionPolicy(self)
        self.filter_policy = FilterPolicy(self)
        self.scorers = ScorerCollection(self)
        self._logger = logger()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Configuration):
            return False
        for key, setting in vars(self).items():
            if isinstance(setting, (int, float, str, bool, list)):
                if (
                    vars(self)[key] != vars(other)[key]
                    or self.search != other.search
                    or self.post_processing != other.post_processing
                ):
                    return False
        return True

    @classmethod
    def from_dict(cls, source: StrDict) -> "Configuration":
        """
        Loads a configuration from a dictionary structure.
        The parameters not set in the dictionary are taken from the default values.
        The policies and stocks specified are directly loaded.

        :param source: the dictionary source
        :return: a Configuration object with settings from the source
        """
        expansion_config = source.pop("expansion", {})
        filter_config = source.pop("filter", {})
        stock_config = source.pop("stock", {})
        scorer_config = source.pop("scorer", {})

        # template_search_config = source.pop("template_search", {})
        # if template_search_config:
        #     config_obj.template_search = FastTemplateSearchEngine(
        #         csv_path=template_search_config.get("csv_path", ""),
        #         openai_key=template_search_config.get("openai_key", None)
        #     )

        config_obj = Configuration()
        config_obj._update_from_config(dict(source))

        config_obj.expansion_policy.load_from_config(**expansion_config)
        config_obj.filter_policy.load_from_config(**filter_config)
        config_obj.stock.load_from_config(**stock_config)
        config_obj.scorers.create_default_scorers()
        config_obj.scorers.load_from_config(**scorer_config)

        # if isinstance(config_obj.search.steer, dict):
        #     config_obj.search.steer = _SteerConfiguration(**config_obj.search.steer)

        if isinstance(config_obj.search.llm_guidance, dict):
            config_obj.search.llm_guidance = _LLMGuidanceConfiguration(
                **config_obj.search.llm_guidance
            )

        return config_obj

    @classmethod
    def from_file(cls, filename: str) -> "Configuration":
        """
        Loads a configuration from a yaml file.
        The parameters not set in the yaml file are taken from the default values.
        The policies and stocks specified in the yaml file are directly loaded.
        The parameters in the yaml file may also contain environment variables as
        values.

        :param filename: the path to a yaml file
        :return: a Configuration object with settings from the yaml file
        :raises:
            ValueError: if parameter's value expects an environment variable that
                does not exist in the current environment
        """
        with open(filename, "r") as fileobj:
            txt = fileobj.read()
        environ_var = re.findall(r"\$\{.+?\}", txt)
        for item in environ_var:
            if item[2:-1] not in os.environ:
                raise ValueError(f"'{item[2:-1]}' not in environment variables")
            txt = txt.replace(item, os.environ[item[2:-1]])
        _config = yaml.load(txt, Loader=yaml.SafeLoader)
        return Configuration.from_dict(_config)

    def _update_from_config(self, config: StrDict) -> None:
        self.post_processing = _PostprocessingConfiguration(
            **config.pop("post_processing", {})
        )

        search_config = config.pop("search", {})
        for setting, value in search_config.items():
            if value is None:
                continue
            if not hasattr(self.search, setting):
                raise AttributeError(f"Could not find attribute to set: {setting}")
            if setting.endswith("_bonds"):
                if not isinstance(value, list):
                    raise ValueError("Bond settings need to be lists")
                value = _handle_bond_pair_tuples(value) if value else []
            if setting == "algorithm_config":
                if not isinstance(value, dict):
                    raise ValueError("algorithm_config settings need to be dictionary")
                self.search.algorithm_config.update(value)
            else:
                setattr(self.search, setting, value)

        template_search_config = config.pop("template_search", {})
        if template_search_config:
            self.template_search = FastTemplateSearchEngine(
                csv_with_embeddings=template_search_config.get("csv_path", ""),
                openai_key=template_search_config.get("openai_key", None),
            )

        route_validator_config = config.pop("route_validator", {})
        if route_validator_config:
            self.route_validator = RouteValidator(
                model=route_validator_config["model"],
                temperature=route_validator_config["temperature"],
                backend=route_validator_config.get("backend", "openrouter"),
            )


def _handle_bond_pair_tuples(bonds: List[List[int]]) -> List[List[int]]:
    if not all(len(bond_pair) == 2 for bond_pair in bonds):
        raise ValueError("Lists of bond pairs to break/freeze should be of length 2")
    return [bond_pair[:2] for bond_pair in bonds]
