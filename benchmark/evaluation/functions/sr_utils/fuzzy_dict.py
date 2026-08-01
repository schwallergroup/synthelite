import json
from collections import UserDict
from difflib import SequenceMatcher


class FuzzyDict(UserDict):
    """A dictionary that allows for fuzzy string matching on keys.

    This dictionary can retrieve values using keys that are similar to,
    but not exactly the same as, the stored keys. The similarity threshold
    determines how close a match needs to be.
    """

    def __init__(self, items=None, threshold=0.6):
        """Initialize a FuzzyDict with optional items and threshold.

        Args:
            items: Optional dictionary or list of key-value pairs to initialize with
            threshold: Float between 0 and 1, minimum similarity ratio to consider a match
        """
        super().__init__()
        self.threshold = threshold

        if items:
            if isinstance(items, dict):
                self.update(items)
            else:
                for key, value in items:
                    self[key] = value

    def _get_similarity(self, s1, s2):
        """Calculate string similarity between s1 and s2."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def __getitem__(self, key):
        """Get an item using fuzzy matching if exact key is not found."""
        try:
            # First try exact match
            return self.data[key]
        except KeyError:
            # If not found, try fuzzy matching
            if not isinstance(key, str):
                raise KeyError(key)

            # Find best match above threshold
            best_match = None
            best_ratio = self.threshold - 0.001  # Use slightly less than threshold for comparison

            for k in self.data:
                if not isinstance(k, str):
                    continue

                ratio = self._get_similarity(key, k)
                if ratio >= self.threshold and ratio > best_ratio:
                    best_match = k
                    best_ratio = ratio

            if best_match is None:
                raise KeyError(f"No fuzzy match found for '{key}' with threshold {self.threshold}")

            return self.data[best_match]

    def get_with_match(self, key):
        """Get both the value and the matched key."""
        try:
            # First try exact match
            return key, self.data[key]
        except KeyError:
            # If not found, try fuzzy matching
            if not isinstance(key, str):
                raise KeyError(key)

            # Find best match above threshold
            best_match = None
            best_ratio = self.threshold - 0.001

            for k in self.data:
                if not isinstance(k, str):
                    continue

                ratio = self._get_similarity(key, k)
                if ratio >= self.threshold and ratio > best_ratio:
                    best_match = k
                    best_ratio = ratio

            if best_match is None:
                raise KeyError(f"No fuzzy match found for '{key}' with threshold {self.threshold}")

            return best_match, self.data[best_match]

    def set_threshold(self, threshold):
        """Set the similarity threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold

    def get_matching_keys(self, key, include_ratios=False):
        """Get all keys that match above the threshold.

        Args:
            key: The key to match against
            include_ratios: If True, return (key, ratio) tuples instead of just keys

        Returns:
            A list of matching keys or (key, ratio) tuples, sorted by decreasing similarity
        """
        if not isinstance(key, str):
            return []

        matches = []

        for k in self.data:
            if not isinstance(k, str):
                continue

            ratio = self._get_similarity(key, k)
            if ratio >= self.threshold:
                if include_ratios:
                    matches.append((k, ratio))
                else:
                    matches.append(k)

        # Sort by decreasing similarity
        if include_ratios:
            matches.sort(key=lambda x: x[1], reverse=True)
        else:
            matches.sort(
                key=lambda k: self._get_similarity(key, k), reverse=True
            )

        return matches

    @classmethod
    def from_json(
        cls,
        file_path,
        key_field,
        value_field,
        threshold=0.6,
        filter_field=None,
        filter_value=None,
    ):
        """Load a FuzzyDict from a JSON file in jsonlines format with customizable key and value fields.

        Args:
            file_path: Path to the JSON file
            key_field: The field name to use as the key in the dictionary
            value_field: The field name to use as the value in the dictionary
            threshold: Similarity threshold for fuzzy matching
            filter_field: Optional field name to filter entries by
            filter_value: Optional value to match against filter_field

        Returns:
            A new FuzzyDict instance loaded with data from the JSON file where values are lists
            containing all values associated with each key.
        """
        data = {}

        with open(file_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                if not line.strip():
                    continue

                # Parse the JSON line
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filter if specified
                if filter_field is not None and filter_value is not None:
                    if (
                        filter_field not in entry
                        or entry[filter_field] != filter_value
                    ):
                        continue

                # Extract key and value if they exist
                if key_field in entry and value_field in entry:
                    key = entry[key_field]
                    value = entry[value_field]

                    # Only add if key is a valid string
                    if isinstance(key, str) and key:
                        # Initialize list if key is new, otherwise append to existing list
                        if key not in data:
                            data[key] = [value]
                        else:
                            data[key].append(value)

        # Create and return a new FuzzyDict with the loaded data
        return cls(data, threshold=threshold)


# Test the fix with your example
if __name__ == "__main__":
    # Test case to demonstrate the fix
    fd = FuzzyDict(threshold=0.4)
    fd["Mitsunobu aryl ether"] = ["some_smarts_pattern"]
    
    print(f"Stored key: 'Mitsunobu aryl ether'")
    print(f"Threshold: {fd.threshold}")
    
    # Calculate similarity
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, "mitsunobu aryl", "mitsunobu aryl ether").ratio()
    print(f"Similarity between 'Mitsunobu aryl' and 'Mitsunobu aryl ether': {similarity:.3f}")
    
    try:
        result = fd["Mitsunobu aryl"]
        print(f"Result for 'Mitsunobu aryl': {result}")
    except KeyError as e:
        print(f"KeyError: {e}")
        
    # Also test the get_with_match method
    try:
        matched_key, value = fd.get_with_match("Mitsunobu aryl")
        print(f"Matched key: '{matched_key}', Value: {value}")
    except KeyError as e:
        print(f"KeyError in get_with_match: {e}")