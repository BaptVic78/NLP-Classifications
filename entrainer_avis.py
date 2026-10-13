"""Les étapes du notebook IMDB, dans leur ordre de lecture."""
from pathlib import Path
import pandas as pd
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nettoyage import nettoyer_avis

racine = Path(__file__).resolve().parent

# 1. Charger et retirer les doublons.
movies = pd.read_csv(racine / 'comments-movies-classification' / 'IMDB Dataset.csv')
movies = movies.drop_duplicates().dropna().reset_index(drop=True)

# 2. Nettoyer : lettres, minuscules, stopwords NLTK et stemming.
dossier_nltk = racine / 'demo' / 'data' / 'nltk'
nltk.data.path.insert(0, str(dossier_nltk))
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=str(dossier_nltk), raise_on_error=True)
    stop_words = set(nltk.corpus.stopwords.words('english'))
corpus = []
for review in movies['review']:
    corpus.append(nettoyer_avis(review, stop_words))
y = movies['sentiment']

# 3. Séparer AVANT TF-IDF pour éviter de connaître le vocabulaire du test.
text_train, text_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=42)

# 4. Vectoriser avec les 5000 variables du notebook.
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(text_train)
X_test = tfidf.transform(text_test)

# 5. Entraîner chaque modèle explicitement.
model_logistic_regression = LogisticRegression(max_iter=1000)
model_logistic_regression.fit(X_train, y_train)

model_naives_bayes = MultinomialNB()
model_naives_bayes.fit(X_train, y_train)

# 6. Évaluer les modèles sur les mêmes avis de test.
modeles = {'Régression logistique': model_logistic_regression,
           'Naive Bayes': model_naives_bayes}
scores = {}
matrices = {}
for nom, modele in modeles.items():
    y_pred = modele.predict(X_test)
    scores[nom] = accuracy_score(y_test, y_pred)
    matrices[nom] = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])
    print(nom, scores[nom], flush=True)
    print(classification_report(y_test, y_pred), flush=True)

# 7. Sauvegarder pour l'interface (aucun entraînement à chaque clic).
sortie = racine / 'demo' / 'artifacts'
sortie.mkdir(parents=True, exist_ok=True)
joblib.dump({'tfidf': tfidf, 'modeles': modeles, 'scores': scores, 'matrices': matrices,
             'classes': ['negative', 'positive'], 'stop_words': stop_words,
             'train_size': len(y_train), 'test_size': len(y_test)}, sortie / 'movies_simple.joblib')
