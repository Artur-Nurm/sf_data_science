import warnings

from models.encoder import Encoder


class Model:
    def __init__(self, model=None) -> None:
        # to encode input into format that required for model prediction
        self.encoder = Encoder()
        # takes as input a model with predict and predict_proba methods
        self.model = model

    def get_features_dict(self):
        """
        Returns dict that contains every feature that must be encoded
        and the order of encoding
        """
        return self.encoder.get_features_dict()

    def predict(self, X, class_names=True):
        """
        Predict class baed on the input feature vector

        input: X - feature vector, can be list or dict
               if X is a dict, returns only one class for one record
               if X is a list, function returns many classes - one for each record
        returns: predicted class and a corresponding probability
        """
        if self.model is None:
            warnings.warn("Model is not defined! Load the model first")
            return

        # encode input into format that required for model prediction
        if isinstance(X, dict):
            encoded_X = self.encoder.encode_one_record(X).reshape((1, -1))
        elif isinstance(X, list):
            encoded_X = self.encoder.encode_records(X)
        else:
            raise ValueError("X must be list or dict")

        # prediction
        try:
            y = self.model.predict(encoded_X)
            p = self.model.predict_proba(encoded_X)
            if y[0] == 2:
                p = p[0][2]
            else:
                p = p[0][:2].sum()

        except AttributeError:
            raise AttributeError("Model must have predict and predict_proba methods")

        if class_names:
            y = [self.encoder.outcome_mapping[i] for i in y]

        return y, p
