from dataclasses import dataclass
import json

@dataclass
class Dataset:
    data: list[dict|tuple|list]

    @classmethod
    def from_yaml(cls, path):
        examples = []
        i = 0
        with open(path, 'r') as f:
            while(True):
                line = f.readline()
                i += 1
                if line == '': 
                    break 
                if line.strip() != '': # Ignores empty lines in between
                    examples.append(json.loads(line))
        return cls(data=examples)

