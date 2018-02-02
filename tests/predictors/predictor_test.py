# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import my_library

class TestPaperClassifierPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
            "title": "Interferring Discourse Relations in Context",
            "paperAbstract": (
                    "We investigate various contextual effects on text "
                    "interpretation, and account for them by providing "
                    "contextual constraints in a logical theory of text "
                    "interpretation. On the basis of the way these constraints "
                    "interact with the other knowledge sources, we draw some "
                    "general conclusions about the role of domain-specific "
                    "information, top-down and bottom-up discourse information "
                    "flow, and the usefulness of formalisation in discourse theory."
            )
        }

        archive = load_archive('tests/fixtures/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'paper-classifier')

        result = predictor.predict_json(inputs)

        label = result.get("label")
        assert label in ['AI', 'ML', 'ACL']

        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)
