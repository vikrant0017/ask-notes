from dataclasses import dataclass
import json

@dataclass
class Dataset:
    data: list[dict|tuple|list]

    @classmethod
    def from_jsonl(cls, path):
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

    def rename_columns(self, mapping, strict=False):
        """
        Rename the columns of the dataset based on the provided mapping.

        This method updates the dataset's internal data structure by renaming the keys of each 
        dictionary in the dataset according to the specified mapping. If a key in the dataset 
        does not exist in the mapping, it will retain its original name. 

        Parameters:
        - mapping: dict[str,str] - A dictionary where keys are the original column names and values are 
          the new column names.
        - strict: bool - If True, raises a KeyError for any key in the dataset that is not 
          present in the mapping. If False, such keys will be left unchanged.

        Returns:
        None
        """
        for row in self.data:
            for cur_col_name, new_col_name in mapping.items():
                """If the key associated with the column to rename is not in the row, then either
                throw error or continue with the next key to rename, based on the strict paramter flag"""
                if cur_col_name not in row:
                    if strict:
                        raise KeyError(f"Key '{cur_col_name}' not found in mapping.")
                    else:
                        continue

                value = row.pop(cur_col_name)
                row[new_col_name] = value

        
    def to_jsonl(self, path):
        """
        Write the dataset to a JSON Lines (jsonl) file.

        Each item in the dataset is serialized as a JSON object and written to the specified file,
        with each object on a new line.

        Parameters:
        - path: str - The file path where the jsonl data will be written.
        """
        with open(path, 'w') as f:
            for item in self.data:
                f.write(json.dumps(item) + '\n')
