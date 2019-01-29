from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('paper-classifier')
class PaperClassifierPredictor(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        title = json_dict['title']
        abstract = json_dict['paperAbstract']
        instance = self._dataset_reader.text_to_instance(title=title, abstract=abstract)

        return instance
