from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('paper-classifier')
class PaperClassifierPredictor(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        # TODO: get title and abstract out of json_dict

        # TODO: create an instance
        # thing that might be useful:
        #  * self._dataset_reader.text_to_instance

        # TODO: get the mapping label_id -> label_name
        # thing that might be useful:
        #  * self._model.vocab.get_index_to_token_vocabulary

        # TODO: convert it to a list of label_names all_labels
        # [label0_name, label1_name, label2_name]
        return instance, {"all_labels": all_labels}
