"""
Compute ROUGE metric.

For BioASQ.py-QA task b, summary questions,
ROUGE (and BLEU) metrics are used to evaluate the quality of the summaries provided by the systems.
They measure the overlap between the words in the system-generated summaries
and a set of reference summaries.


"""

from rouge import Rouge

hypothesis = "the cat was found under the bed"
reference = "the cat was under the bed"

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)

print(scores)


# This will give you a dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores,
# each with 'f' (F1-score),
# 'p' (precision), and 'r' (recall).

