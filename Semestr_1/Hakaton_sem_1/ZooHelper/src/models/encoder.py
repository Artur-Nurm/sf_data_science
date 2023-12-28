import numpy as np
import warnings


class Encoder:
    """
    class to encode input dict that contained features from self.encode_order
    into format that required for model prediction
    """

    def __init__(self) -> None:
        # mapping for target feature that will be predicted for the model
        self.outcome_mapping = {0: "died", 1: "euthanized", 2: "lived"}
        # every encoded string will have that size
        self.encoded_len = 89
        # contains every feature that must be encoded and the order of encoding
        self.encode_order = {
            "rectal_temp": ["numeric"],
            "pulse": ["numeric"],
            "respiratory_rate": ["numeric"],
            "nasogastric_reflux_ph": ["numeric"],
            "packed_cell_volume": ["numeric"],
            "total_protein": ["numeric"],
            "abdomo_protein": ["numeric"],
            "abdomen": ["distend_large", "distend_small", "firm", "normal", "other"],
            "abdominal_distention": ["moderate", "none", "severe", "slight"],
            "abdomo_appearance": ["clear", "cloudy", "serosanguious"],
            "age": ["adult", "young"],
            "capillary_refill_time": ["3", "less_3_sec", "more_3_sec"],
            "cp_data": ["no", "yes"],
            "mucous_membrane": [
                "bright_pink",
                "bright_red",
                "dark_cyanotic",
                "normal_pink",
                "pale_cyanotic",
                "pale_pink",
            ],
            "nasogastric_reflux": ["less_1_liter", "more_1_liter", "none"],
            "nasogastric_tube": ["none", "significant", "slight"],
            "pain": ["alert", "depressed", "extreme_pain", "mild_pain", "severe_pain"],
            "peripheral_pulse": ["absent", "increased", "normal", "reduced"],
            "peristalsis": ["absent", "hypermotile", "hypomotile", "normal"],
            "rectal_exam_feces": ["absent", "decreased", "increased", "normal"],
            "surgery": ["no", "yes"],
            "surgical_lesion": ["no", "yes"],
            "temp_of_extremities": ["cold", "cool", "normal", "warm"],
            "lesion_1": [
                "gastric",
                "sm_intestine",
                "lg_colon",
                "lg_colon_and_cecum",
                "cecum",
                "transverse_colon",
                "retum/descending_colon",
                "uterus",
                "bladder",
                "all_intestinal_sites",
                "simple",
                "strangulation",
                "inflammation",
                "other",
                "mechanical",
                "paralytic",
                "obturation",
                "intrinsic",
                "extrinsic",
                "adynamic",
                "volvulus/torsion",
                "intussuption",
                "thromboembolic",
                "hernia",
                "lipoma/slenic_incarceration",
                "displacement",
            ],
        }

    # for futher integration with the bot
    def get_features_dict(self):
        """
        Returns dict that contains every feature that must be encoded
        and the order of encoding
        """
        return self.encode_order

    def encode_records(self, records: list) -> np.array:
        """
        Encodes multiple records

        input: list of dicts, where every dict is a record,
            that must be encoded
        returns: array of encoded records
        """
        encoded_records = None
        # enumerating and encoding records one by one
        for record in records:
            encoded = self.encode_one_record(record)

            if encoded_records is None:
                encoded_records = encoded
                continue

            encoded_records = np.vstack((encoded_records, encoded))
        return encoded_records

    def encode_one_record(self, record: dict) -> np.array:
        """
        Encodes dict into array required for model as input

        input: dict with features:
               record = {
                    feature1: value1,
                    feature2: value2, ...
               }
        returns: encoded recorded in the array format
                [encoded_value1] + [encoded_value2] + ...
        """

        encoded_record = np.array([])
        # enumerate self.encode_order items for maintaining the order
        # in which the array is filled
        for key, possible_values in self.encode_order.items():
            # getting a value corresponding to the feature
            actual_value = record.get(key)

            # if there is no such feature in the input dict
            if actual_value is None:
                encoded_value = self.__encode_null_elem(key, possible_values)
            # if the value of the feature should be numeric
            elif possible_values[0] == "numeric":
                encoded_value = self.__encode_numeric_elem(
                    key, possible_values, actual_value
                )
            # for lesion_1 feature there are special encoding rules
            elif key == "lesion_1":
                encoded_value = self.__encode_lesion(
                    key, possible_values, str(actual_value)
                )
            # if the value of the feature should be categorical
            else:
                encoded_value = self.__encode_categorical_elem(
                    key, possible_values, actual_value
                )

            # Checkpoint: the encoded_value array must contain at least 1 value
            assert len(encoded_value) != 0, "Something went wrong, {key} wast encoded."

            # adding new i encoded value to encoded_record array with i-1 encoded values
            encoded_record = np.append(encoded_record, encoded_value)

        # Checkpoint: the array with encoded_values must have 89 elements
        #             at the end of the function
        assert (
            len(encoded_record) == self.encoded_len
        ), "Something went wrong, try another input"
        return encoded_record

    def __encode_null_elem(self, key: str, possible_values: list) -> np.array:
        """
        Encodes feature that was missed in the input record

        input: key - name of the missing feature
               possible_values - list of the values that can match this feature
        returns: array with default encoded value corresponding to this feature
        """

        # if the feature was missed, we will use this values instead
        # each value corresponds to normal indicators of horse health
        default_values = {
            "rectal_temp": [37.8],
            "pulse": [73],
            "respiratory_rate": [9],
            "nasogastric_reflux_ph": [4],
            "packed_cell_volume": [None],
            "total_protein": [7],
            "abdomo_protein": [2],
            "abdomen": [0, 0, 0, 1, 0],
            "abdominal_distention": [0, 1, 0, 0],
            "abdomo_appearance": [None] * 3,
            "age": [1, 0],
            "capillary_refill_time": [None] * 3,
            "cp_data": [1, 0],
            "mucous_membrane": [0, 0, 0, 1, 0, 0],
            "nasogastric_reflux": [1, 0, 0],
            "nasogastric_tube": [0, 0, 1],
            "pain": [1, 0, 0, 0, 0],
            "peripheral_pulse": [0, 0, 1, 0],
            "peristalsis": [0, 0, 0, 1],
            "rectal_exam_feces": [None] * 4,
            "surgery": [1, 0],
            "surgical_lesion": [1, 0],
            "temp_of_extremities": [None] * 4,
            "lesion_1": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
        warnings.warn(f"{key}: Value was missed.")
        encoded = np.array(default_values[key])
        return encoded

    def __encode_numeric_elem(
        self, key: str, possible_values: list, actual_value: str
    ) -> np.array:
        """
        Encodes feature must have numerica value

        input: key - name of the numeric feature
               possible_values - always [numeric]
               actual_value - value from input record
        returns: array with 1 numeric value
        """
        # value must have a float conversion
        # if it can't be converted, then the input was incorrect
        try:
            encoded = np.array([float(actual_value)])
        except ValueError:
            warnings.warn(f"Wrong format, {key} must be numeric! Value was skiped.")
            encoded = np.zeros(len(possible_values)).astype(np.float16)
        return encoded

    def __encode_categorical_elem(
        self, key: str, posible_values: list, actual_value: str
    ) -> np.array:
        """
        One hot encoding for categorical feature

        input: key - name of the missing feature
               possible_values - list of the values that can match this feature
               actual_value - value from input record
        returns: array with 1 numeric value

        Example:
                possible_values: [normal, not_normal, smth]
                actual_value: not_normal
                returns: [0, 1, 0]
        """

        posible_values = np.array(posible_values)
        # creates mask of possible_values that has 1 at the position
        # where posible_values element matches the actual_value and 0 at the other
        encoded = np.in1d(posible_values, actual_value).astype(np.float16)

        # if the actual_value can not be found in the posible_values array
        if not any(encoded):
            warnings.warn(f" {key}: Wrong format, value was skiped.")
        return encoded

    def __encode_lesion(
        self, key: str, possible_values: list, actual_value: str
    ) -> np.array:
        """
        Encodes lesion_1 feature

        input: key - always lesion_1
               possible_values - list with details of the lesion
               actual_value - numeric code with 4-5 digits in which the details of the lesion are encoded
        returns: array with encoded lesion
        """

        posible_values = np.array(possible_values)
        # decoding numeric code into multiple lesion features
        relevant_values = np.array(self.__split_lesion(actual_value))
        # one hot encoding for lesion features
        encoded = np.in1d(posible_values, relevant_values).astype(np.float16)
        # if actual_value was in wrong format
        if not any(encoded):
            warnings.warn(f" {key}: Wrong format, value was skiped.")
        return encoded

    def __split_lesion(self, code: str) -> list:
        """
        Decode lesion code into lesion features

        input: code - numeric code with 4-5 digits in which the details of the lesion are encoded
                      the first digit encodes the site of lesion
                      the second digit encodes the type of lesion
                      the third digit encodes the subtype of lesion
                      the fourth digit encodes the lesion specific code
        returns: decoded numeric code - list of features

        Example:
             input: 2209
             returns: [sm_intestine, strangulation, none, lipoma/slenic_incarceration]
        """
        # first digit of the input code
        lesion_site = {
            "1": "gastric",
            "2": "sm_intestine",
            "3": "lg_colon",
            "4": "lg_colon_and_cecum",
            "5": "cecum",
            "6": "transverse_colon",
            "7": "retum/descending_colon",
            "8": "uterus",
            "9": "bladder",
            "11": "all_intestinal_sites",
            "00": "none",
        }
        # second digit of the input code
        lesion_type = {
            "1": "simple",
            "2": "strangulation",
            "3": "inflammation",
            "4": "other",
            "0": "none",
        }
        # third digit of the input code
        lesion_subtype = {
            "1": "mechanical",
            "2": "paralytic",
            "0": "none",
        }
        # fourth digit of the input code
        lesion_specific_code = {
            "1": "obturation",
            "2": "intrinsic",
            "3": "extrinsic",
            "4": "adynamic",
            "5": "volvulus/torsion",
            "6": "intussuption",
            "7": "thromboembolic",
            "8": "hernia",
            "9": "lipoma/slenic_incarceration",
            "10": "displacement",
            "0": "none",
        }
        if code[:2] in ["00", "11"]:
            site, type, subtype, s_code = code[:2], code[2], code[3], code[4:]
        elif code[0] == "0":
            site, type, subtype, s_code = "none", "none", "none", "none"
        else:
            site, type, subtype, s_code = code[0], code[1], code[2], code[3:]

        return [
            lesion_site.get(site, "none"),
            lesion_type.get(type, "none"),
            lesion_subtype.get(subtype, "none"),
            lesion_specific_code.get(s_code, "none"),
        ]
