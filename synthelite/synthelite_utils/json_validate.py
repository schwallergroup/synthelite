import json
import jsonschema
from jsonschema import validate

DATA_DIR = "synthelite/synthelite_utils/"


def check_json(json_data):
    with open(DATA_DIR + "data/JSONSCHEMA.txt") as f:
        json_schema = json.load(f)

    try:
        validate(instance=json_data, schema=json_schema)
        return True
    except jsonschema.exceptions.ValidationError as err:
        print(err)
        return False


if __name__ == "__main__":

    with open(DATA_DIR + "data/example_output_sc.json") as f:
        data = json.load(f)

    print(check_json(data))
