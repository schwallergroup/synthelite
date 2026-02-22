import json
import os
import re

import numpy as np
import requests
from pydantic import BaseModel
from rdkit import Chem
from typing import Dict, List


class TranslateReaxys(BaseModel):
    url: str = os.environ.get(
        "TRANSLATOR_URL", "http://localhost:5000/translator/translate"
    )
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    def __call__(self, smiles: str | List[str]):
        if isinstance(smiles, list):
            return [self(smi) for smi in smiles]
        else:
            src = self.tokenize_input(smiles)

        data = {"src": src, "id": 0}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps([data])
        )
        out = self.get_response(smiles, response)
        return out

    def get_response(self, smi_input, response):
        resp = response.json()[0][0]
        smiles = re.sub(" ", "", resp["tgt"])
        score = np.exp(resp["pred_score"])
        return TranslateReaxys.wrap_syntree(
            reactants=smiles, product=smi_input, score=score
        )

    @staticmethod
    def wrap_syntree(reactants: str, product: str, score: float):
        d = {
            "type": "mol",
            "hide": False,
            "smiles": product,
            "is_chemical": True,
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "hide": False,
                    "smiles": f"{reactants}>>{product}",
                    "is_reaction": True,
                    "metadata": {
                        "template_hash": "onmt",
                        "classification": "onmt",
                        "library_occurence": 0,
                        "policy_probability": score,
                        "policy_probability_rank": 0,
                        "policy_name": "uspto",
                        "template_code": 0,
                        "template": "onmt",
                        "mapped_reaction_smiles": f"{reactants}>>{product}",
                    },
                    "children": [
                        TranslateReaxys.make_mol(r) for r in reactants.split(".")
                    ],
                }
            ],
        }
        return d

    @staticmethod
    def make_mol(smiles: str):
        return {
            "type": "mol",
            "hide": False,
            "smiles": smiles,
            "is_chemical": True,
            "in_stock": False,
        }

    def tokenize_input(self, smiles: str):
        SMI_REGEX_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        smiles_regex = re.compile(SMI_REGEX_PATTERN)
        canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        tokens = [token for token in smiles_regex.findall(canonical_smiles)]
        return " ".join(tokens)


if __name__ == "__main__":
    tr = TranslateReaxys()
    print(
        tr(
            [
                "O=C(OCC)[C@@H](N1C[C@@H](C(F)(F)F)NC1=O)C2=CN=NC(N)=C2",
                "CC(C)C1=CC=C(C=C1)C(=O)OCC1=CC=CC=C1",
            ]
        )
    )
