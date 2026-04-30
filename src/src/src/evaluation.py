from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def compute_bleu(reference, prediction):
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    return sentence_bleu([reference_tokens], prediction_tokens)


def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure


def hallucination_rate(prediction, context):
    pred_words = set(prediction.split())
    context_words = set(context.split())

    hallucinated = pred_words - context_words

    if len(pred_words) == 0:
        return 0

    return len(hallucinated) / len(pred_words)


def clinical_f1(reference, prediction):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    common = set(ref_tokens) & set(pred_tokens)

    if len(common) == 0:
        return 0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * (precision * recall) / (precision + recall)
