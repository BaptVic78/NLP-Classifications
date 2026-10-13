"""Charger, nettoyer, vectoriser, prédire : interface du projet Avis de films."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from nettoyage import nettoyer_avis

racine = Path(__file__).resolve().parent
st.set_page_config(page_title='Avis de films', layout='wide')
st.title('Analyse des avis de films')
st.write('Classification des avis IMDB en positif ou négatif.')
nom_fichier = 'movies_simple.joblib'

if not (racine / 'demo' / 'artifacts' / nom_fichier).exists():
    st.info('Lancez bash train.sh pour préparer les modèles.')
    st.stop()
donnees = joblib.load(racine / 'demo' / 'artifacts' / nom_fichier)
nom_modele = st.sidebar.selectbox('Modèle', list(donnees['modeles']))
modele = donnees['modeles'][nom_modele]
exemple = 'The acting was brilliant and the story was wonderful.'
texte = st.text_area('Commentaire en anglais', exemple, height=130)

if st.button('Prédire', type='primary'):
    texte_nettoye = nettoyer_avis(texte, donnees['stop_words'])
    X = donnees['tfidf'].transform([texte_nettoye])
    if not texte_nettoye or X.nnz == 0:
        st.warning('Aucun mot reconnu. Essayez un autre texte en anglais.')
    else:
        prediction = modele.predict(X)[0]
        st.success(f'Prédiction : {prediction}')
        st.caption('Texte après nettoyage')
        st.code(texte_nettoye, language=None)

st.subheader('Résultats sur le jeu de test')
st.dataframe(pd.DataFrame({'Modèle': list(donnees['scores']), 'Accuracy': list(donnees['scores'].values())}), hide_index=True)
st.write(f"{donnees['train_size']} textes d'entraînement · {donnees['test_size']} textes de test")
st.subheader('Matrice de confusion')
st.caption('Lignes : classes réelles. Colonnes : classes prédites.')
st.dataframe(pd.DataFrame(donnees['matrices'][nom_modele], index=donnees['classes'], columns=donnees['classes']))
st.caption('Avis en anglais uniquement. Modèles : régression logistique et Naive Bayes.')
