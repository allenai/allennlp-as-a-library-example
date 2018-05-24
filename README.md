A simple example for how to build your own model using AllenNLP as a dependency.  An explanation
of all of the code in this repository is given in the [part 1](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/using_as_a_library_pt1.md) and [part 2](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/using_as_a_library_pt2.md) of the AllenNLP
tutorial.

There are two main pieces of code you need to write in order to make a new model: a
`DatasetReader` and a `Model`.  In this repository, we constructed a `DatasetReader` for reading
academic papers formatted as a JSON lines file (you can see an example of the data in
[`tests/fixtures/s2_papers.jsonl`](tests/fixtures/s2_papers.jsonl)).  We then constructed a model
to classify the papers given some label (which we specified as the paper's venue in the
`DatasetReader`).  Finally, we added a script to use AllenNLP's training commands from a
third-party repository, and an experiment configuration for running a real model on real data.

To train this model, after setting up your development environment by running
`pip install -r requirements.txt`, you run:

```bash
python -m allennlp.run train experiments/venue_classifier.json -s /tmp/your_output_dir_here --include-package my_library
```

This example was written by the AllenNLP team.  You can see a similar example repository written
by others [here](https://github.com/recognai/get_started_with_deep_learning_for_text_with_allennlp).
