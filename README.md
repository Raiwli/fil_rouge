# Projet fil rouge — Lille-Lesquin (SYNOP 2023)

Projet complet des séances 1 à 4 en Python, utilisant le CSV Météo-France fourni.

## Lancement

1. Installer Python 3.10 ou plus récent.
2. Dans ce dossier : `python -m pip install -r requirements.txt`
3. Exécuter : `python projet_meteo.py`

Le programme réécrit les dossiers `resultats/` et `figures/` sans accès Internet.

## Contenu

- `projet_meteo.py` : code complet commenté, fonctions, validations, analyses et régressions.
- `CORRIGE_COMPLET.md` : résolution pas à pas, réponses aux questions, interprétation chiffrée, guide oral.
- `donnees/synop_lille_2023.csv` : CSV brut original fourni.
- `Fil rouge 1.pdf` à `Fil rouge 4.pdf` : énoncés originaux.
- `resultats/donnees_nettoyees.csv` : données filtrées/converties.
- `resultats/agregation_mensuelle.csv` : températures, vent, humidité et cumuls de pluie.
- `resultats/agregation_horaire.csv` : moyennes par heure.
- `resultats/dates_manquantes.csv`, `rapport_valeurs_manquantes.csv`, `manquants_par_mois.csv`, `diagnostic.txt` : qualité des données.
- `resultats/correlations.csv` : corrélations de Pearson.
- `resultats/comparaison_modeles.csv` : scores test/train, baseline, tailles et périodes.
- `resultats/coefficients_*.csv` : coefficients des trois variantes de régression.
- `resultats/predictions_test.csv` et `residus_par_*.csv` : prédictions et biais temporels.
- `figures/` : 12 figures PNG, dont 8 graphiques exploratoires et 4 de modélisation.

## Précautions méthodologiques

- Les précipitations `rr3 = -0.1` signalent une trace, comptée à zéro pour les cumuls, avec valeur brute conservée. Les valeurs absentes ne sont pas remplacées par zéro.
- La séparation apprentissage/test respecte l'ordre chronologique des mesures (80/20).
- Les températures estimées à partir de mesures météorologiques contemporaines ne constituent pas une prévision anticipée.
- Le R² test peut être négatif ; les résultats sont reproduits sans ajustement artificiel.
- L'extension électrique RTE reste non calculée : aucun fichier éCO2mix n'a été fourni.
