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

        instance2 = {"title": ["GRASPER", ":", "A", "Permissive", "Planning", "Robot"],
                     "abstract": ["Execut", "ion", "of", "classical", "plans"],
                     "venue": "AI"}

        instance3 = {"title": ["Route", "Planning", "under", "Uncertainty", ":", "The", "Canadian",
                               "Traveller", "Problem"],
                     "abstract": ["The", "Canadian", "Traveller", "problem", "is"],
                     "venue": "AI"}

        # check that there are 10 instances

        # check that the first instance agrees with instance1

        # assert len(instances) == 10
        # fields = instances[0].fields
        # assert [t.text for t in fields["title"].tokens] == instance1["title"]
        # assert [t.text for t in fields["abstract"].tokens[:5]] == instance1["abstract"]
        # assert fields["label"].label == instance1["venue"]
        # fields = instances[1].fields
        # assert [t.text for t in fields["title"].tokens] == instance2["title"]
        # assert [t.text for t in fields["abstract"].tokens[:5]] == instance2["abstract"]
        # assert fields["label"].label == instance2["venue"]
        # fields = instances[2].fields
        # assert [t.text for t in fields["title"].tokens] == instance3["title"]
        # assert [t.text for t in fields["abstract"].tokens[:5]] == instance3["abstract"]
        # assert fields["label"].label == instance3["venue"]
