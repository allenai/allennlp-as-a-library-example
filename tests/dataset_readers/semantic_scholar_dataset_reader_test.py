# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from my_library.dataset_readers import SemanticScholarDatasetReader

class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = SemanticScholarDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/s2_papers.jsonl'))

        instance1 = {"title": ["Interferring", "Discourse", "Relations", "in", "Context"],
                     "abstract": ["We", "investigate", "various", "contextual", "effects"],
                     "venue": "ACL"}

        # TODO: check that there are 10 instances

        # TODO: check that the first instance agrees with instance1
