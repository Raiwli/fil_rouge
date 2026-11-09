"""Projet fil rouge météo Lille-Lesquin 2023 — séances 1 à 4.

Entrée : donnees/synop_lille_2023.csv (SYNOP, ;, 'mq', dates UTC).
Sorties : resultats/ et figures/. Exécution : python projet_meteo.py
Choix : la cible est temperature_c. Les variables contemporaines u, ff,
pression_mer_hpa et indicateurs calendaires sont évaluées. Le calendrier seul
fournit un modèle de référence plus compatible avec une prévision anticipée.
Le modèle principal est une estimation à l'instant du relevé, et NON une
prévision météorologique à horizon futur (les covariables de l'instant ne sont
pas forcément connues à l'avance). Séparation chronologique 80/20 sans fuite.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'donnees' / 'synop_lille_2023.csv'
RESULTATS = BASE / 'resultats'
FIGURES = BASE / 'figures'
STATION = 7015
TEST_RATIO = 0.20
COLONNES = ['numer_sta','date','t','pmer','pres','u','dd','ff','rr3']
NUMERIQUES = ['t','pmer','pres','u','dd','ff','rr3']
FEATURES = ['u','ff','pression_mer_hpa','mois_sin','mois_cos','heure_sin','heure_cos']
FEATURES_CALENDRIER = ['mois_sin','mois_cos','heure_sin','heure_cos']


def charger(path=SOURCE):
    """Lire le fichier CSV et valider la présence des colonnes attendues."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'CSV introuvable : {path}')
    try:
        brut = pd.read_csv(path, sep=';', na_values=['mq'], low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f'Lecture CSV impossible : {exc}') from exc
    absentes = sorted(set(COLONNES) - set(brut.columns))
    if absentes:
        raise ValueError(f'Colonnes attendues absentes : {absentes}')
    if brut.empty:
        raise ValueError('Le CSV est vide.')
    return brut


def nettoyer(brut):
    """Convertir les unités et dates, filtrer Lille et préserver les NaN mesurés."""
    df = brut[COLONNES].copy()
    df['date'] = pd.to_datetime(df['date'].astype('string'), format='%Y%m%d%H%M%S', errors='coerce', utc=True)
    df['numer_sta'] = pd.to_numeric(df['numer_sta'], errors='coerce')
    for c in NUMERIQUES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['date','numer_sta'])
    df = df.loc[df['numer_sta'].eq(STATION)].sort_values('date').drop_duplicates('date').copy()
    if df.empty:
        raise ValueError('Aucune observation pour la station 07015 avec une date valide.')
    df['temperature_c'] = df['t'] - 273.15
    # Repère de plausibilité volontairement large, pour ne pas masquer les anomalies.
    hors_plausible = df['temperature_c'].notna() & ~df['temperature_c'].between(-60, 60)
    if hors_plausible.any():
        raise ValueError(f'{int(hors_plausible.sum())} températures hors [-60 ; 60] °C : vérifier les unités et les données.')
    df['pression_mer_hpa'] = df['pmer']/100
    df['pression_station_hpa'] = df['pres']/100
    # -0.1 mm = trace de précipitation ; 0 pour la somme, NaN reste NaN.
    df['precipitations_3h_mm'] = df['rr3'].mask(df['rr3'] == -0.1, 0)
    negatives = df['precipitations_3h_mm'].lt(0)
    if negatives.any():
        # Autres codes négatifs inconnus : ne pas les compter comme de la pluie.
        df.loc[negatives, 'precipitations_3h_mm'] = np.nan
    df['annee'] = df['date'].dt.year
    df['mois'] = df['date'].dt.month
    df['heure_utc'] = df['date'].dt.hour
    df['mois_sin'] = np.sin(2*np.pi*(df['mois']-1)/12)
    df['mois_cos'] = np.cos(2*np.pi*(df['mois']-1)/12)
    df['heure_sin'] = np.sin(2*np.pi*df['heure_utc']/24)
    df['heure_cos'] = np.cos(2*np.pi*df['heure_utc']/24)
    return df.reset_index(drop=True)


def exporter_diagnostic(brut, df):
    """Exporter disponibilité, fréquences et statistiques de qualité de données."""
    manque = pd.DataFrame({'colonne': NUMERIQUES,
                           'manquantes': [int(df[c].isna().sum()) for c in NUMERIQUES],
                           'taux_pourcent': [round(100*df[c].isna().mean(),2) for c in NUMERIQUES]})
    manque.to_csv(RESULTATS/'rapport_valeurs_manquantes.csv', index=False)
    grille = pd.date_range('2023-01-01', '2023-12-31 21:00', freq='3h', tz='UTC')
    manquantes = grille.difference(pd.DatetimeIndex(df['date']))
    pd.DataFrame({'date_manquante_utc': manquantes}).to_csv(RESULTATS/'dates_manquantes.csv', index=False)
    msg = [f'Station : {STATION:05d} ; période : 2023 ; lignes brutes : {len(brut)}',
           f'Lignes station conservées : {len(df)} / {len(grille)} attendues ; absentes : {len(manquantes)}',
           f'Températures min/max : {df.temperature_c.min():.2f} / {df.temperature_c.max():.2f} °C',
           f'rr3 : min={df.rr3.min():.2f}, max={df.rr3.max():.2f}, '
           f'zéros={(df.rr3 == 0).mean():.1%}, traces={(df.rr3 == -0.1).sum()}, '
           f'manquantes={df.rr3.isna().sum()}',
           'Les traces (-0.1) sont converties à 0 pour les cumuls ; les NaN restent manquants.',
           'Pas de remplissage artificiel ; agrégations basées sur les observations disponibles.',
           'Types bruts :', brut[COLONNES].dtypes.to_string(), '', 'Valeurs manquantes :', manque.to_string(index=False)]
    (RESULTATS/'diagnostic.txt').write_text('\n'.join(msg)+'\n', encoding='utf-8')
    return msg


def agreger(df):
    """Produire les statistiques par mois, par heure et sur les absences mensuelles."""
    mensuel = df.groupby(['annee','mois'], as_index=False).agg(
        temperature_moyenne_c=('temperature_c','mean'), temperature_min_c=('temperature_c','min'),
        temperature_max_c=('temperature_c','max'), nb_mesures_temperature=('temperature_c','count'),
        humidite_moyenne_pct=('u','mean'), vent_moyen_ms=('ff','mean'),
        precipitations_total_mm=('precipitations_3h_mm','sum'),
        nb_mesures_precipitations=('precipitations_3h_mm','count'))
    mensuel.to_csv(RESULTATS/'agregation_mensuelle.csv', index=False)
    df.groupby('heure_utc', as_index=False).agg(temperature_moyenne_c=('temperature_c','mean'), nb_mesures=('temperature_c','count')).to_csv(RESULTATS/'agregation_horaire.csv', index=False)
    taux = df.assign(manquant_rr3=df.rr3.isna()).groupby(['annee','mois'],as_index=False).agg(
        nb_precipitations_manquantes=('manquant_rr3','sum'), nb_releves=('date','size'))
    taux['pourcentage_manquant'] = 100*taux.nb_precipitations_manquantes/taux.nb_releves
    taux.to_csv(RESULTATS/'manquants_par_mois.csv', index=False)
    return mensuel


def sauvegarder(fig, nom):
    """Enregistrer une figure lisible avant fermeture."""
    fig.tight_layout()
    fig.savefig(FIGURES/nom, dpi=150, bbox_inches='tight')
    plt.close(fig)


def explorer(df, mensuel):
    """Créer les graphiques exploratoires et matrices de corrélation."""
    fig, ax = plt.subplots(figsize=(12,4)); ax.plot(df.date, df.temperature_c, lw=.65)
    ax.set(title='Lille-Lesquin : température par relevé (2023)', xlabel='Date UTC', ylabel='Température (°C)'); ax.grid(alpha=.2)
    sauvegarder(fig,'01_temperature_releves.png')
    fig, ax = plt.subplots(figsize=(9,4)); ax.plot(mensuel.mois,mensuel.temperature_moyenne_c,'o-',label='Moyenne')
    ax.fill_between(mensuel.mois,mensuel.temperature_min_c,mensuel.temperature_max_c,alpha=.17,label='Extrêmes observés')
    ax.set(title='Température mensuelle : moyenne et extrêmes', xlabel='Mois',ylabel='Température (°C)',xticks=range(1,13)); ax.legend(); ax.grid(alpha=.2)
    sauvegarder(fig,'02_temperature_mensuelle.png')
    # Distribution complète : les traces (-0.1) sont enregistrées comme 0 mm.
    pluie = df.precipitations_3h_mm.dropna()
    fig, ax = plt.subplots(figsize=(9,4)); ax.hist(pluie,bins=35,edgecolor='white')
    ax.set(title='Précipitations sur 3 heures : distribution (avec jours secs)',xlabel='Précipitations (mm / 3 h)',ylabel='Nombre de relevés')
    sauvegarder(fig,'03_precipitations_distribution.png')
    fig, ax = plt.subplots(figsize=(9,4)); ax.hist(pluie[pluie>0],bins=25,edgecolor='white')
    ax.set(title='Précipitations positives uniquement',xlabel='Précipitations (mm / 3 h)',ylabel='Nombre de relevés')
    sauvegarder(fig,'04_precipitations_positives.png')
    fig, ax = plt.subplots(figsize=(9,4)); ax.bar(mensuel.mois,mensuel.precipitations_total_mm)
    ax.set(title='Cumuls mensuels observés (hors périodes sans mesure)',xlabel='Mois',ylabel='Précipitations cumulées (mm)',xticks=range(1,13))
    sauvegarder(fig,'05_precipitations_mensuelles.png')
    correl = df[['temperature_c','pression_mer_hpa','pression_station_hpa','u','ff']].corr()
    correl.to_csv(RESULTATS/'correlations.csv')
    fig, ax = plt.subplots(figsize=(8,5)); im=ax.imshow(correl.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(correl)),correl.columns,rotation=35,ha='right'); ax.set_yticks(range(len(correl)),correl.index)
    for i in range(len(correl)):
        for j in range(len(correl)):
            ax.text(j,i,f'{correl.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    ax.set_title('Corrélations de Pearson'); fig.colorbar(im,ax=ax)
    sauvegarder(fig,'06_correlations.png')
    cor_temp = correl['temperature_c'].drop('temperature_c').dropna()
    principale = cor_temp.abs().idxmax()
    fig, ax = plt.subplots(figsize=(9,5)); ax.scatter(df[principale],df.temperature_c,s=7,alpha=.3)
    ax.set(title=f'Température et {principale} (r={cor_temp[principale]:.3f})',xlabel=principale,ylabel='Température (°C)')
    sauvegarder(fig,'07_nuage_correlation.png')
    fig,ax=plt.subplots(figsize=(10,5)); df.boxplot(column='temperature_c',by='mois',ax=ax)
    fig.suptitle('');ax.set(title='Température par mois',xlabel='Mois',ylabel='Température (°C)')
    sauvegarder(fig,'08_boxplot_mensuel.png')
    return correl,principale


def preparer_xy(df, variables):
    """Valider et aligner sans NaN X, y et horodatages ; renvoyer pertes."""
    inconnues = sorted(set(variables+['temperature_c','date'])-set(df.columns))
    if inconnues:
        raise ValueError(f'Variables demandées absentes : {inconnues}')
    travail = df[['date','temperature_c']+variables].copy().sort_values('date')
    for col in ['temperature_c']+variables:
        travail[col] = pd.to_numeric(travail[col],errors='coerce')
    travail = travail.replace([np.inf,-np.inf],np.nan).dropna()
    if len(travail)<30:
        raise ValueError(f'Données insuffisantes pour régression : {len(travail)} lignes complètes (minimum 30).')
    return travail, len(df)-len(travail)


def scores(y, predit):
    """Calculer MAE, RMSE et R² en degrés Celsius et sans arrondi prématuré."""
    return {'MAE_C':float(mean_absolute_error(y,predit)),
            'RMSE_C':float(np.sqrt(mean_squared_error(y,predit))),
            'R2':float(r2_score(y,predit))}


def ajuster(df,variables,nom,graphiques=False):
    """Entraîner une régression sur les 80 % premiers relevés et évaluer les 20 % suivants."""
    travail, perdus = preparer_xy(df,variables)
    coupure = int(len(travail)*(1-TEST_RATIO))
    train,test = travail.iloc[:coupure],travail.iloc[coupure:]
    X_train,y_train=train[variables],train.temperature_c
    X_test,y_test=test[variables],test.temperature_c
    modele=LinearRegression().fit(X_train,y_train)
    predit_train=modele.predict(X_train); predit_test=modele.predict(X_test)
    # Baseline déterministe : température moyenne du seul ensemble d'entraînement.
    ref=np.full(len(y_test),float(y_train.mean()))
    ligne={'modele':nom,'n_total':len(travail),'lignes_perdues':perdus,
           'n_entrainement':len(train),'n_test':len(test),
           'train_debut':str(train.date.iloc[0]),'train_fin':str(train.date.iloc[-1]),
           'test_debut':str(test.date.iloc[0]),'test_fin':str(test.date.iloc[-1])}
    for prefix,y,p in [('train',y_train,predit_train),('test',y_test,predit_test),('reference_test',y_test,ref)]:
        ligne.update({prefix+'_'+k:v for k,v in scores(y,p).items()})
    pd.DataFrame({'variable':variables,'coefficient':modele.coef_}).to_csv(RESULTATS/f'coefficients_{nom}.csv',index=False)
    if graphiques:
        test_sortie=test[['date','temperature_c','heure_utc','mois']].copy() if {'heure_utc','mois'}.issubset(test.columns) else test[['date','temperature_c']].copy()
        test_sortie['prediction_c']=predit_test
        test_sortie['erreur_c']=test_sortie.temperature_c-test_sortie.prediction_c
        test_sortie.to_csv(RESULTATS/'predictions_test.csv',index=False)
        fig,ax=plt.subplots(figsize=(11,4)); ax.plot(test.date,y_test,label='Réel',lw=1);ax.plot(test.date,predit_test,label='Prédit',lw=1,alpha=.8)
        ax.set(title='Test chronologique : températures observées vs estimées',xlabel='Date UTC',ylabel='Température (°C)');ax.legend()
        sauvegarder(fig,'09_predictions_test.png')
        fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(y_test,predit_test,alpha=.45,s=12)
        bounds=[min(y_test.min(),predit_test.min()),max(y_test.max(),predit_test.max())]
        ax.plot(bounds,bounds,'k--',lw=1);ax.set(title='Prédiction contre réalité (test)',xlabel='Observé (°C)',ylabel='Prédit (°C)')
        sauvegarder(fig,'10_reel_vs_predit.png')
        fig,ax=plt.subplots(figsize=(9,4)); ax.scatter(test.date,y_test-predit_test,s=9,alpha=.55);ax.axhline(0,color='black',ls='--')
        ax.set(title='Erreur = température réelle - estimée (test)',xlabel='Date UTC',ylabel='Résidu (°C)')
        sauvegarder(fig,'11_residus_test.png')
        # Le mois et l'heure servent au diagnostic ; pas à entraîner sur le test.
        diagnostic = pd.DataFrame({'date':test.date,'mois':test.date.dt.month,'heure_utc':test.date.dt.hour,'erreur_c':y_test-predit_test})
        saison = diagnostic.groupby('mois').erreur_c.agg(['mean','count']); saison.to_csv(RESULTATS/'residus_par_mois.csv')
        heure=diagnostic.groupby('heure_utc').erreur_c.agg(['mean','count']);heure.to_csv(RESULTATS/'residus_par_heure.csv')
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(heure.index,heure['mean']);ax.axhline(0,color='black',lw=.8)
        ax.set(title='Erreur signée moyenne par heure UTC (test)',xlabel='Heure UTC',ylabel='Erreur moyenne (°C)')
        sauvegarder(fig,'12_residus_par_heure.png')
    return ligne,modele


def rapport(df,mensuel,correl,variable_principale,tableau):
    """Rédiger automatiquement la résolution argumentée avec les chiffres réellement calculés."""
    pluie=df.precipitations_3h_mm
    plus_pluvieux=mensuel.loc[mensuel.precipitations_total_mm.idxmax()]
    meilleur=correl['temperature_c'].drop('temperature_c').abs().idxmax()
    principal=tableau.loc[tableau.modele=='principal'].iloc[0]
    cal=tableau.loc[tableau.modele=='calendrier'].iloc[0]
    sans=tableau.loc[tableau.modele=='sans_humidite'].iloc[0]
    s=f'''# Corrigé du projet fil rouge — séances 1 à 4

## Séance 1 — cadrage

La séance 1 est une présentation, et ne contient pas d'exercice à rendre. Station Lille-Lesquin (07015), période 01/01/2023–31/12/2023, SYNOP toutes les 3 heures.

## Séance 2 — chargement, nettoyage, première analyse

1. Le CSV est séparé par `;` ; `mq` signifie valeur manquante. {len(df)} relevés valides ont été conservés pour 2023 contre 2920 théoriques (soit {2920-len(df)} absences de créneaux). Vérifier les `dtypes` après `read_csv` ; `mq` peut faire apparaître des colonnes numériques comme chaînes si non traité.
2. Les variables utiles sont date, température (`t`), pressions (`pmer`, `pres`), humidité (`u`), vitesse (`ff`), direction (`dd`) et précipitations sur 3 h (`rr3`).
3. Conversion `temperature_c = t - 273.15`, sans écraser `t`. Min = {df.temperature_c.min():.2f} °C ; max = {df.temperature_c.max():.2f} °C. Les pressions sont aussi converties de Pa en hPa.
4. On supprime les dates invalides, mais on ne fabrique pas des mesures météo. On conserve les NaN pour les statistiques pandas, qui ignorent les valeurs absentes ; seules les lignes incomplètes sur les variables du modèle sont écartées avant la régression. Pour `rr3`, la trace codée -0.1 est comptabilisée comme 0 mm dans les cumuls ; les valeurs `mq` demeurent NaN.
5. Les moyennes, minima, maxima et nombres de mesures sont dans `resultats/agregation_mensuelle.csv`. La saisonnalité est visible dans `figures/02_temperature_mensuelle.png`.
6. La commande `python projet_meteo.py` recrée intégralement les fichiers. Extension : agrégation horaire et manque de précipitation par mois.

## Séance 3 — exploration et structuration

1. Le tracé des {len(df)} observations révèle les variations de quelques heures et les épisodes ponctuels ; la moyenne mensuelle permet une lecture des saisons mais masque ces fluctuations. Les 2 graphes indiquent leurs axes et unités. Les créneaux absents sont répertoriés mais non interpolés.
2. `rr3` brut : min {df.rr3.min():.2f}, max {df.rr3.max():.2f} mm ; proportion de zéro strict {df.rr3.eq(0).mean():.1%} ; traces (-0.1) : {df.rr3.eq(-.1).sum()} ; valeurs manquantes : {df.rr3.isna().sum()}. Un histogramme représente la distribution des quantités sur 3 heures ; la barre des zéros domine généralement. Un second histogramme des seuls relevés positifs révèle mieux l'intensité de pluie. Un diagramme en barres des **sommes mensuelles**, plutôt que des moyennes, répond au mois le plus pluvieux. Ici : mois {int(plus_pluvieux.mois)} avec {plus_pluvieux.precipitations_total_mm:.2f} mm **cumulés sur les mesures disponibles** ; comparer les mois requiert de tenir compte des relevés manquants.
3. Corrélations dans `resultats/correlations.csv`. La plus forte corrélation absolue hors cible est `{meilleur}` (r = {correl.loc[meilleur,'temperature_c']:.3f}). Le nuage de points montre la dispersion ; une corrélation n'est ni une causalité ni une garantie de prédiction. Les observations successives sont temporellement dépendantes, ce qui crée des structures saisonnières.
4. Toutes les figures sont enregistrées dans `figures/` avec `savefig`, avant `close`.
5. Les étapes sont réparties dans `charger`, `nettoyer`, `agreger`, `explorer`, `preparer_xy`, `ajuster`, `rapport` et `main`, avec docstrings.
6. Fichier absent : `FileNotFoundError` explicite ; colonnes absentes, données vides, station absente, températures aberrantes, nombre de lignes insuffisant : `ValueError` explicites.
7. Une nouvelle exécution régénère tous les résultats sans dépendre d'un état interactif.

**Note de choix des variables pour la séance 4 :** L'humidité, la vitesse du vent et la pression au niveau de la mer sont des covariables observées susceptibles d'aider à estimer la température au même instant ; les corrélations seules ne suffisent pas à établir leur utilité conditionnelle. On ajoute des composantes cycliques du mois et de l'heure pour représenter les variations annuelles et diurnes sans rupture artificielle entre décembre/janvier et 23 h/0 h. On évite le point de rosée `td`, trop étroitement associé à la température et susceptible de constituer une fuite indirecte selon le cas d'usage.

## Séance 4 — régression et interprétation

1. **Cible** `temperature_c` ; **variables explicatives** {', '.join(FEATURES)}. Les pressions ne sont pas une transformation directe de la température cible, mais mesurées au même instant. **Limite essentielle :** ce modèle estime la température à partir de variables contemporaines ; ce n'est pas une vraie prévision future sans accès futur à ces covariables.
2. On convertit toutes les variables en numérique et élimine seulement les lignes incomplètes des colonnes nécessaires au modèle. Modèle principal : {int(principal.n_total)} lignes utiles, {int(principal.lignes_perdues)} éliminées ; X a {len(FEATURES)} colonnes et y une valeur par ligne, index aligné.
3. **Découpage chronologique** : 80 % entraînement, 20 % test ; {int(principal.n_entrainement)} / {int(principal.n_test)} lignes. Entraînement du {principal.train_debut} au {principal.train_fin} ; test du {principal.test_debut} au {principal.test_fin}. Un tirage aléatoire rapprocherait artificiellement les observations voisines dans train et test ; le découpage chronologique mesure mieux un transfert vers une période ultérieure.
4. `LinearRegression.fit()` n'est appliqué qu'au jeu d'entraînement. L'intercept est l'ordonnée à l'origine et chaque coefficient représente l'évolution estimée de y pour +1 unité de sa variable à autres variables constantes. Les coefficients dépendent de l'unité et de la colinéarité : leur grandeur brute n'est pas directement comparable.
5. **Résultats modèle principal sur test :** MAE = {principal.test_MAE_C:.3f} °C ; RMSE = {principal.test_RMSE_C:.3f} °C ; R² = {principal.test_R2:.3f}. Sur entraînement : MAE = {principal.train_MAE_C:.3f} °C ; R² = {principal.train_R2:.3f}. Référence naïve (moyenne de la température train, sans fuite) sur test : MAE = {principal.reference_test_MAE_C:.3f} °C ; R² = {principal.reference_test_R2:.3f}. L'erreur absolue moyenne exprime l'écart moyen en degrés ; le RMSE accentue l'impact des grosses erreurs. Un écart important entre train et test suggère une généralisation limitée, des changements saisonniers ou une spécification imparfaite.
6. Les figures `09` à `12` comparent observations/prédictions et erreurs sur test ; les CSV `residus_par_heure.csv` et `residus_par_mois.csv` indiquent d'éventuels biais systématiques. Variante **sans humidité** : MAE test = {sans.test_MAE_C:.3f} °C, R² = {sans.test_R2:.3f} ; variante **calendrier uniquement** : MAE test = {cal.test_MAE_C:.3f} °C, R² = {cal.test_R2:.3f}. Ces variantes mesurent l'apport des covariables et réduisent les risques de dépendre de mesures indisponibles au moment de prévoir.
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
'''
    (BASE/'CORRIGE_COMPLET.md').write_text(s,encoding='utf-8')


def main():
    """Exécuter intégralement le projet et sauvegarder tous les livrables."""
    RESULTATS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    brut=charger(); df=nettoyer(brut)
    lignes=exporter_diagnostic(brut,df)
    df.to_csv(RESULTATS/'donnees_nettoyees.csv',index=False)
    mensuel=agreger(df)
    correl,principale=explorer(df,mensuel)
    variants=[('principal',FEATURES,True),('sans_humidite',[x for x in FEATURES if x!='u'],False),('calendrier',FEATURES_CALENDRIER,False)]
    scores_variants=[]
    for nom,features,fig in variants:
        ligne,_=ajuster(df,features,nom,fig)
        scores_variants.append(ligne)
    tableau=pd.DataFrame(scores_variants)
    tableau.to_csv(RESULTATS/'comparaison_modeles.csv',index=False)
    rapport(df,mensuel,correl,principale,tableau)
    resume='\n'.join(lignes)+'\n\nCOMPARAISON DES MODÈLES :\n'+tableau[['modele','n_total','lignes_perdues','test_MAE_C','test_RMSE_C','test_R2','reference_test_MAE_C']].to_string(index=False)+'\n'
    (RESULTATS/'execution_console.txt').write_text(resume,encoding='utf-8')
    print(resume)
    print('Projet terminé. Résultats :',RESULTATS,'; figures :',FIGURES)


if __name__=='__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f'ERREUR : {exc}',file=sys.stderr)
        sys.exit(1)
