from transformers import AutoTokenizer, AutoModelForQuestionAnswering, \
    pipeline

# model source: https://huggingface.co/savasy/bert-base-turkish-squad
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")
predictor = pipeline("question-answering", model=model, tokenizer=tokenizer)


def infer(question: str,
          context: str) -> str:
    """
    Gets the best answer for a question in a given context. (in turkish)
    :param question: question for the context
    :param context: Current context
    :return: best answer
    """
    results = predictor(context=context,
                        question=question,
                        max_answer_len=100,
                        top_k=10,
                        handle_impossible_answer=True)
    return results[0]['answer']

"""
another context: balıkların en besleyici kısmı kemikleri ve kabuğudur, kılçıkları ve kabukları ile yenen tek balık hamsidir, daha lezzetli çok balık vardır ama en besleyici balık hamsidir.
another question: Kılçıkları yenen tek balık nedir?
"""
if __name__ == '__main__':
    context = "ABASIYANIK, Sait Faik. Hikayeci (Adapazarı 23 Kasım 1906-İstanbul 11 Mayıs 1954). \
İlk öğrenimine Adapazarı’nda Rehber-i Terakki Mektebi’nde başladı. İki yıl kadar Adapazarı İdadisi’nde okudu.\
İstanbul Erkek Lisesi’nde devam ettiği orta öğrenimini Bursa Lisesi’nde tamamladı (1928). İstanbul Edebiyat \
Fakültesi’ne iki yıl devam ettikten sonra babasının isteği üzerine iktisat öğrenimi için İsviçre’ye gitti. \
Kısa süre sonra iktisat öğrenimini bırakarak Lozan’dan Grenoble’a geçti. Üç yıl başıboş bir edebiyat öğrenimi \
gördükten sonra babası tarafından geri çağrıldı (1933). Bir müddet Halıcıoğlu Ermeni Yetim Mektebi'nde Türkçe \
gurup dersleri öğretmenliği yaptı. Ticarete atıldıysa da tutunamadı. Bir ay Haber gazetesinde adliye muhabirliği\
yaptı (1942). Babasının ölümü üzerine aileden kalan emlakin geliri ile avare bir hayata başladı. Evlenemedi.\
Yazları Burgaz adasındaki köşklerinde, kışları Şişli’deki apartmanlarında annesi ile beraber geçen bu fazla \
içkili bohem hayatı ömrünün sonuna kadar sürdü."
    question = "Ne zaman avare bir hayata başladı?"
    res = infer(context=context, question=question)
    print(res)
