import pandas as pd
from pydantic import BaseModel, root_validator
from typing import Optional, List


class ConditionRecommendTemplate(BaseModel):
    dreagents: Optional[pd.DataFrame | str] = None

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, hasht: str) -> List[str]:
        try:
            rlist = pd.Series(self.dreagents[hasht])
            pred = rlist.value_counts().sort_values(ascending=False)
            if pred.index[0] == "":
                pred = pred[1:]
            if len(pred) > 5:
                return pred.head(5).index.tolist()
            return pred.index.tolist()
        except:
            # Do something
            return ["no conditions yet"]

    @staticmethod
    def load_df_reagents(
        rfile: str = "synthelite/data/complete_cond_translated.csv",
    ) -> pd.DataFrame:
        try:
            df = pd.read_csv(rfile, sep="\t", on_bad_lines="skip", low_memory=False)
            df["reagents"] = df["reagents"]
            gdf = df.groupby("TemplateHash")["reagents"].apply(list)
            return gdf.to_dict()
        except pd.errors.ParserError as e:
            print(f"Error reading TSV file: {e}")
            return pd.DataFrame()
        except KeyError as e:
            print(f"Key error: {e}")
            return pd.DataFrame()

    @root_validator(pre=False, skip_on_failure=True)
    def set_df(cls, values):
        """Initialize df for retrieval."""
        if values["dreagents"] is None:
            values["dreagents"] = cls.load_df_reagents()
        elif isinstance(values["dreagents"], str):
            values["dreagents"] = cls.load_df_reagents(values["dreagents"])
        return values
