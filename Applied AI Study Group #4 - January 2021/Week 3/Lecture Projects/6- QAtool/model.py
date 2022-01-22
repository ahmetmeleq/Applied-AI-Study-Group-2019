from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")
predictor=pipeline("question-answering", model=model, tokenizer=tokenizer)

def infer(question,context):
    results = predictor(context=context, question=question, max_answer_len=100, topk=10, handle_impossible_answer=True)
    return results[0]['answer']
