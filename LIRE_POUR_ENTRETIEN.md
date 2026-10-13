# Classification des avis de films — IMDB

Classer un avis de film en anglais comme positif ou négatif avec TF-IDF, régression logistique et Naive Bayes.

## Lancer sous Ubuntu

```bash
cd /home/victor/NLP/NLP-Classifications
bash run.sh
```

Ouvrir http://localhost:8501. Si le port est occupé par une autre instance : `PORT=8511 bash run.sh`.

Pour installer les dépendances : `bash install.sh`. Pour réentraîner : `bash train.sh`.

## Code à expliquer

- `comments-movies-classification/training.ipynb` : notebook de référence, conservé.
- `entrainer_avis.py` : chargement IMDB, dédoublonnage, nettoyage, split train/test, TF-IDF, entraînement et évaluation.
- `nettoyage.py` : minuscules, retrait des balises/caractères non alphabétiques, stopwords NLTK et stemming Porter.
- `app.py` : chargement du modèle, nettoyage d’un nouvel avis, vectorisation et prédiction.

Le TF-IDF apprend uniquement sur le train. Les modèles utilisés par l’interface sont sauvegardés dans `demo/artifacts/movies_simple.joblib`. Le SVM n’est pas utilisé. Les anciens exports ne sont pas chargés.

L’interface affiche la prédiction, le texte nettoyé, l’accuracy et la matrice de confusion. Le dataset Émotions et ses fichiers ont été archivés séparément et ne font plus partie de cette démonstration.
