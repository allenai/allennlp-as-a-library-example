A simple example for how to build your own model using AllenNLP as a dependency.  This README is
not very polished, but we will make this into a tutorial posted on the [AllenNLP
website](http://allennlp.org) soon.

There are two main pieces of code you need to write in order to make a new model: a
`DatasetReader` and a `Model`.  In this repository, we constructed a `DatasetReader` for reading
academic papers formatted as a JSON lines file (you can see an example of the data in
[`tests/fixtures/s2_papers.jsonl`](tests/fixtures/s2_papers.jsonl)).  We then constructed a model
to classify the papers given some label (which we specified as the paper's venue in the
`DatasetReader`).  Finally, we added a script to use AllenNLP's training commands from a
third-party repository, and an experiment configuration for running a real model on real data.

This example was written by the AllenNLP team.  You can see a similar example repository written
by others [here](https://github.com/recognai/get_started_with_deep_learning_for_text_with_allennlp).
