import pandas as pd
import re
import json
from pydantic import BaseModel, Field, root_validator
from typing import Optional, Dict


class TemplateLookup(BaseModel):
    classification_table: Optional[pd.DataFrame] = None
    scalability_table: Optional[Dict] = None
    classes: Optional[Dict] = None
    super_classes: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=False, skip_on_failure=True)
    def load_data(cls, values):
        prefix = "synthelite/data/scalability/"
        values["classification_table"] = pd.read_csv(
            prefix + "uspto_combined_template_lookup.csv", sep="\t"
        )
        values["scalability_table"] = json.load(open(prefix + "scalability_table.json"))
        values["classes"] = json.load(open(prefix + "classes.json"))
        values["super_classes"] = json.load(open(prefix + "super_classes.json"))
        return values

    def remove_pattern(self, input_string):
        pattern = r"\d*\.\d+"
        new_string = re.sub(pattern, "", input_string)
        return new_string.strip().lower()

    def lookup(self, retro_template: str):
        try:
            reaction_class = self.classification_table.loc[
                self.classification_table["retro_template"] == retro_template,
                "classification",
            ].iloc[0]
        except IndexError:
            reaction_class = "other"
        classes = reaction_class.split(".")
        clas = ".".join(classes[:2])
        super_class = classes[0]

        reaction_class = self.remove_pattern(reaction_class)

        if reaction_class in self.scalability_table:
            scalability = str(self.scalability_table[reaction_class])
        elif clas in self.classes:
            scalability = str(self.classes[clas])
        elif super_class in self.super_classes:
            scalability = str(self.super_classes[super_class])
        else:
            scalability = "4"

        return scalability
