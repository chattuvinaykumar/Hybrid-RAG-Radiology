from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def compute_bleu(reference, prediction):
    return sentence_bleu([reference.split()], prediction.split())


def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)['rougeL'].fmeasure


def hallucination_rate(prediction, context):
    pred_words = set(prediction.split())
    context_words = set(context.split())

    if len(pred_words) == 0:
        return 0

    hallucinated = pred_words - context_words
    return len(hallucinated) / len(pred_words)
