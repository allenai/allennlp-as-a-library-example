# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase

from my_library.dataset_readers import SemanticScholarDatasetReader


class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = SemanticScholarDatasetReader()
        dataset = reader.read('tests/fixtures/s2_papers.jsonl')

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

        assert len(dataset.instances) == 10
        fields = dataset.instances[0].fields
        assert [t.text for t in fields["title"].tokens] == instance1["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance1["abstract"]
        assert fields["label"].label == instance1["venue"]
        fields = dataset.instances[1].fields
        assert [t.text for t in fields["title"].tokens] == instance2["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance2["abstract"]
        assert fields["label"].label == instance2["venue"]
        fields = dataset.instances[2].fields
        assert [t.text for t in fields["title"].tokens] == instance3["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance3["abstract"]
        assert fields["label"].label == instance3["venue"]
