# Corrigé du projet fil rouge — séances 1 à 4

## Séance 1 — cadrage

La séance 1 est une présentation, et ne contient pas d'exercice à rendre. Station Lille-Lesquin (07015), période 01/01/2023–31/12/2023, SYNOP toutes les 3 heures.

## Séance 2 — chargement, nettoyage, première analyse

1. Le CSV est séparé par `;` ; `mq` signifie valeur manquante. 2864 relevés valides ont été conservés pour 2023 contre 2920 théoriques (soit 56 absences de créneaux). Vérifier les `dtypes` après `read_csv` ; `mq` peut faire apparaître des colonnes numériques comme chaînes si non traité.
2. Les variables utiles sont date, température (`t`), pressions (`pmer`, `pres`), humidité (`u`), vitesse (`ff`), direction (`dd`) et précipitations sur 3 h (`rr3`).
3. Conversion `temperature_c = t - 273.15`, sans écraser `t`. Min = -5.00 °C ; max = 33.90 °C. Les pressions sont aussi converties de Pa en hPa.
4. On supprime les dates invalides, mais on ne fabrique pas des mesures météo. On conserve les NaN pour les statistiques pandas, qui ignorent les valeurs absentes ; seules les lignes incomplètes sur les variables du modèle sont écartées avant la régression. Pour `rr3`, la trace codée -0.1 est comptabilisée comme 0 mm dans les cumuls ; les valeurs `mq` demeurent NaN.
5. Les moyennes, minima, maxima et nombres de mesures sont dans `resultats/agregation_mensuelle.csv`. La saisonnalité est visible dans `figures/02_temperature_mensuelle.png`.
6. La commande `python projet_meteo.py` recrée intégralement les fichiers. Extension : agrégation horaire et manque de précipitation par mois.

## Séance 3 — exploration et structuration

1. Le tracé des 2864 observations révèle les variations de quelques heures et les épisodes ponctuels ; la moyenne mensuelle permet une lecture des saisons mais masque ces fluctuations. Les 2 graphes indiquent leurs axes et unités. Les créneaux absents sont répertoriés mais non interpolés.
2. `rr3` brut : min -0.10, max 19.00 mm ; proportion de zéro strict 72.4% ; traces (-0.1) : 228 ; valeurs manquantes : 5. Un histogramme représente la distribution des quantités sur 3 heures ; la barre des zéros domine généralement. Un second histogramme des seuls relevés positifs révèle mieux l'intensité de pluie. Un diagramme en barres des **sommes mensuelles**, plutôt que des moyennes, répond au mois le plus pluvieux. Ici : mois 11 avec 126.70 mm **cumulés sur les mesures disponibles** ; comparer les mois requiert de tenir compte des relevés manquants.
3. Corrélations dans `resultats/correlations.csv`. La plus forte corrélation absolue hors cible est `u` (r = -0.606). Le nuage de points montre la dispersion ; une corrélation n'est ni une causalité ni une garantie de prédiction. Les observations successives sont temporellement dépendantes, ce qui crée des structures saisonnières.
4. Toutes les figures sont enregistrées dans `figures/` avec `savefig`, avant `close`.
5. Les étapes sont réparties dans `charger`, `nettoyer`, `agreger`, `explorer`, `preparer_xy`, `ajuster`, `rapport` et `main`, avec docstrings.
6. Fichier absent : `FileNotFoundError` explicite ; colonnes absentes, données vides, station absente, températures aberrantes, nombre de lignes insuffisant : `ValueError` explicites.
7. Une nouvelle exécution régénère tous les résultats sans dépendre d'un état interactif.

**Note de choix des variables pour la séance 4 :** L'humidité, la vitesse du vent et la pression au niveau de la mer sont des covariables observées susceptibles d'aider à estimer la température au même instant ; les corrélations seules ne suffisent pas à établir leur utilité conditionnelle. On ajoute des composantes cycliques du mois et de l'heure pour représenter les variations annuelles et diurnes sans rupture artificielle entre décembre/janvier et 23 h/0 h. On évite le point de rosée `td`, trop étroitement associé à la température et susceptible de constituer une fuite indirecte selon le cas d'usage.

## Séance 4 — régression et interprétation

1. **Cible** `temperature_c` ; **variables explicatives** u, ff, pression_mer_hpa, mois_sin, mois_cos, heure_sin, heure_cos. Les pressions ne sont pas une transformation directe de la température cible, mais mesurées au même instant. **Limite essentielle :** ce modèle estime la température à partir de variables contemporaines ; ce n'est pas une vraie prévision future sans accès futur à ces covariables.
2. On convertit toutes les variables en numérique et élimine seulement les lignes incomplètes des colonnes nécessaires au modèle. Modèle principal : 2864 lignes utiles, 0 éliminées ; X a 7 colonnes et y une valeur par ligne, index aligné.
3. **Découpage chronologique** : 80 % entraînement, 20 % test ; 2291 / 573 lignes. Entraînement du 2023-01-01 00:00:00+00:00 au 2023-10-21 06:00:00+00:00 ; test du 2023-10-21 09:00:00+00:00 au 2023-12-31 21:00:00+00:00. Un tirage aléatoire rapprocherait artificiellement les observations voisines dans train et test ; le découpage chronologique mesure mieux un transfert vers une période ultérieure.
4. `LinearRegression.fit()` n'est appliqué qu'au jeu d'entraînement. L'intercept est l'ordonnée à l'origine et chaque coefficient représente l'évolution estimée de y pour +1 unité de sa variable à autres variables constantes. Les coefficients dépendent de l'unité et de la colinéarité : leur grandeur brute n'est pas directement comparable.
5. **Résultats modèle principal sur test :** MAE = 3.158 °C ; RMSE = 3.961 °C ; R² = -0.273. Sur entraînement : MAE = 2.618 °C ; R² = 0.788. Référence naïve (moyenne de la température train, sans fuite) sur test : MAE = 5.122 °C ; R² = -2.067. L'erreur absolue moyenne exprime l'écart moyen en degrés ; le RMSE accentue l'impact des grosses erreurs. Un écart important entre train et test suggère une généralisation limitée, des changements saisonniers ou une spécification imparfaite.
6. Les figures `09` à `12` comparent observations/prédictions et erreurs sur test ; les CSV `residus_par_heure.csv` et `residus_par_mois.csv` indiquent d'éventuels biais systématiques. Variante **sans humidité** : MAE test = 3.554 °C, R² = -0.536 ; variante **calendrier uniquement** : MAE test = 3.497 °C, R² = -0.505. Ces variantes mesurent l'apport des covariables et réduisent les risques de dépendre de mesures indisponibles au moment de prévoir.
7. `main()` orchestre la préparation, l'entraînement, les scores et les graphiques ; aucun modèle n'est entraîné sur le test.
8. Script documenté, paramètres centralisés, données et résultats dans des dossiers dédiés, exécution déterministe et exceptions explicites.
9. **Restitution orale (5 minutes) :** présenter les données et leur nettoyage (1 min), les graphes et corrélations (1 min), le choix des variables et la séparation temporelle (1 min), les scores face à la référence (1 min), puis les limites et la variante (1 min).

### Limites critiques

- Les données de 2023 ne portent que sur une seule station et une seule année : le modèle ne se généralise pas automatiquement à d'autres années, stations ni à des événements météorologiques extrêmes.
- Les corrélations et coefficients ne prouvent aucun lien causal ; la colinéarité peut modifier fortement les coefficients.
- La séparation par date est réaliste pour une évaluation dans le futur mais les covariables météorologiques au temps t ne sont pas nécessairement disponibles à t-1 : **pas de véritable prévision anticipée**.
- Le modèle linéaire simplifie les interactions non linéaires et les dynamiques temporelles ; une bonne MAE sur ce test ne garantit pas la robustesse à long terme.
- Les cumuls de pluie concernent les seules périodes observées et peuvent sous-estimer le total réel si des mesures manquent.

### Extension optionnelle RTE

L'extension consommation électrique exige un **fichier éCO2mix fourni par le formateur** ; aucun fichier RTE n'est présent dans les pièces fournies. Elle n'est donc pas calculable honnêtement ici, sans inventer de données. Une fois le fichier obtenu, agréger météo et consommation par jour, joindre sur une date et utiliser une séparation chronologique pour prédire la consommation.
