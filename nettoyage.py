import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def nettoyer_avis(texte, stop_words):
    texte = re.sub(r'<[^>]*>', ' ', texte)
    texte = re.sub('[^a-zA-Z]', ' ', texte).lower()
    mots = [stemmer.stem(mot) for mot in texte.split() if mot not in stop_words]
    return ' '.join(mots)
