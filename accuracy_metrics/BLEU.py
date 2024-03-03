"""
Compute BLEU metric.

For Datasets.py-QA task b, summary questions,
BLEU (and ROUGE) metrics are used to evaluate the quality of the summaries provided by the systems.
They measure the overlap between the words in the system-generated summaries
and a set of reference summaries.


In the example snippet below:
    `reference` is a list of reference sentences, and
    `candidate` is the generated sentence.

BLEU scores range from 0 to 1, where 1 means a perfect match.
"""

from nltk.translate.bleu_score import sentence_bleu

reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidate = ['this', 'is', 'a', 'test']

score = sentence_bleu(reference, candidate)
print(score)


