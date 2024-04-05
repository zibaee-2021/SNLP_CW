"""
Compute BERTSCore for two sentences.

('BERTScore: Evaluating Text Generation with BERT'. Zhang et al., ICLR 2020).

BERTScore computes a similarity score for each token in the candidate sentence with each token in the reference
sentence. However, instead of exact matches, it computes token similarity using pre-trained BERT contextual embeddings
('BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'. Devlin et al., NAACL-HLT 2019).

BERTScore computes the similarity of two sentences as a sum of cosine similarities between their tokens’ embeddings.


If no GPU, then can run on Colab at https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q
which uses the default English language model `roberta-large`.
"""
from bert_score import score

# cloning the repo because we need to get some example data
# git clone https://github.com/Tiiiger/bert_score.git
# with open("bert_score/example/hyps.txt") as f:
#     cands = [line.strip() for line in f]
#
# with open("bert_score/example/refs.txt") as f:
#     refs = [line.strip() for line in f]
# cands[0]

# When you are running this cell for the first time,
# it will download the BERT model which will take relatively longer.
# P, R, F1 = score(cands, refs, lang="en", verbose=True)
# print(f'F1 {F1}')
# print(f"System level F1 score: {F1.mean():.3f}")
# import matplotlib.pyplot as plt
# plt.hist(F1, bins=20)
# plt.show()
# from bert_score import plot_example
#
# cand = cands[0]
# ref = refs[0]
# plot_example(cand, ref, lang="en")