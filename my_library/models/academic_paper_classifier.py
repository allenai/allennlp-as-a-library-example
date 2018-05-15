from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("paper_classifier")
class AcademicPaperClassifier(Model):
    """
    This ``Model`` performs text classification for an academic paper.  We assume we're given a
    title and an abstract, and we predict some output label.

    The basic model structure: we'll embed the title and the abstract, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    title_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
    abstract_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the abstract to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AcademicPaperClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != title_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            title_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != abstract_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                title: Dict[str, torch.LongTensor],
                abstract: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        title : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        abstract : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # TODO: embed + get mask + encode title
        # things that might be useful:
        #    * self.text_field_embedder
        #    * util.get_text_field_mask
        #    * self.title_encoder (which takes a tensor and a mask)

        # TODO: embed + get mask + encode abstact

        # TODO: concatenate encodings and feedforward to get logits, put them in output dict
        # things that might be useful:
        #    * torch.cat (make sure to specify which dim to cat along)
        #    * self.classifier_feedforward

        # TODO: if label specified, compute loss and add it to output_dict
        # things that might be useful:
        #    * tensor.squeeze()  [the labels have dimension (batch_size, 1)]
        #    * self.loss (takes logits and labels)

        # TODO: if label specified, update metrics
        # things that might be useful:
        #    * tensor.squeeze
        #    * self.metrics (each one takes logits and labels)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        # TODO: compute class probabilities and add them to output_dict as "class_probabilities"
        # things that might be useful:
        #    * the "logits" in the output dict
        #    * F.softmax (make sure to specify which dim to softmax along,
        #                 the logits have shape (batch_size, num_classes))

        # TODO: compute predictions
        # things that might be useful:
        #    * tensor.cpu().data.numpy()  (get data from a variable to numpy array)
        #    * numpy.argmax (make sure to specify which axis to argmax along)

        # TODO: find predicted label[s] and add them as "label"
        # things that might be useful:
        #    * the array of predictions we just computed
        #    * a list comprehension
        #    * self.vocab.get_token_from_index  (make sure to specify a namespace)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AcademicPaperClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        title_encoder = Seq2VecEncoder.from_params(params.pop("title_encoder"))
        abstract_encoder = Seq2VecEncoder.from_params(params.pop("abstract_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   title_encoder=title_encoder,
                   abstract_encoder=abstract_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
