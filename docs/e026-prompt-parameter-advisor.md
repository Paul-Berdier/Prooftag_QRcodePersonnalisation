# E026 — conseiller contextuel prompt → paramètres

## Décision

E026 n'entraîne ni Stable Diffusion ni ControlNet. Il entraîne un modèle tabulaire qui estime le
résultat d'une **recette DiffQRCoder observée** pour un nouveau prompt et un contexte QR. Il
réduit le nombre de générations à essayer, sans se substituer à QR-Verify.

La règle de décision est lexicographique :

1. probabilité calibrée de réussite QR-Verify ;
2. borne basse pénalisée par l'incertitude ;
3. score de tolérance QR-Verify ;
4. notes humaines lorsqu'elles sont suffisamment nombreuses ;
5. HPS v2.1, CLIP-Aesthetic et CLIPScore ;
6. faible saturation et faible durée.

Une très belle image probablement illisible ne peut donc pas dépasser une candidate qui franchit
la porte de scan. L'image produite par la recette retenue doit encore réussir QR-Verify avant
livraison.

## Pourquoi E007 ne suffit plus

`ContextualParameterAdvisor` était une preuve de concept construite avec les anciens taux de
validation. E026 ajoute :

- `quality_qr_verify_any_exact` comme classe principale ;
- `quality_qr_verify_tolerance_score` comme cible continue ;
- CLIP-Aesthetic, CLIPScore, HPS v2.1, durée et saturation comme régressions séparées ;
- calibration isotone des probabilités hors-échantillon ;
- dispersion des arbres comme indicateur d'incertitude ;
- séparation `GroupKFold` par SHA-256 du **texte exact du prompt** ;
- conservation de la configuration complète réellement demandée ;
- refus d'entraînement si le dataset est trop petit ou ne contient pas les deux classes QR.

## Nouveau contrat CSV

Les exports de campagne incluent maintenant :

- `campaign_id`, `campaign_name`, `payload_hash` et `payload_length` ;
- `prompt_text` et `negative_prompt` ;
- `method_configuration_json` ainsi que les sous-objets génération, modèle et outils ;
- la configuration QR et la sortie réellement sélectionnée ;
- toutes les métriques et les notes humaines existantes.

Le payload demeure absent de la base. Seuls son SHA-256 et sa longueur sont conservés. Cela
permet de représenter la densité probable du QR sans persister une URL sensible.

Les anciennes campagnes n'ont pas de longueur de payload. Les anciens CSV peuvent être sauvés en
fournissant leurs textes dans `LEGACY_PROMPT_CATALOG`, mais une configuration ancienne seulement
reconstituite depuis les diagnostics est signalée par l'audit.

## Entrées et cibles

Le modèle ne reçoit jamais une métrique calculée après génération en entrée. Les variables
autorisées sont :

- projection déterministe de l'embedding CLIP du prompt ;
- longueur, nombre de mots et quelques caractéristiques lexicales du prompt ;
- longueur du payload, ECC, version, masque, taille de module et marge ;
- modèle épinglé, variante de sortie et réutilisation Stage 1 ;
- paramètres complets de génération, Stage 2, SRPG et SR-MPGD.

Les métriques calculées après génération sont exclusivement des cibles. Cette séparation évite la
fuite qui consisterait à prédire la réussite à partir d'un diagnostic de l'image déjà produite.

Le seed n'est pas traité comme une grandeur numérique prédictible. Deux entiers voisins ne
produisent pas des images voisines. En production, il faut appliquer les meilleures recettes à
plusieurs seeds appariés, puis valider les images réelles.

## Collecte intégrée et reprise

Le notebook 21 pilote maintenant directement E026W : 300 prompts, quatre familles, 16 recettes,
trois seeds et jusqu'à 14 400 essais. Le kernel Jupyter reste en CPU ; l'API Kubernetes conserve
seule la RTX et exécute les générations. Il n'y a plus de manifestes à télécharger ni de CSV à
réimporter manuellement.

La collecte est découpée en lots de dix prompts. Après chaque lot, son CSV est écrit dans
`/data/e026-week/<plan-id>/exports`. Toute transition est enregistrée atomiquement dans
`state.json`, et les événements affichés sont aussi ajoutés à `notebook-progress.jsonl`.
Lors d'un arrêt propre, le runner annule uniquement le lot courant et exporte également ses
lignes déjà terminées ; une reprise ne refait donc jamais les lots antérieurs.

Pour lancer :

1. déployer les deux images du même commit avec `bash scripts/deploy-e026-notebook.sh` ;
2. ouvrir `21_e026_prompt_parameter_advisor.ipynb` ;
3. définir l'URL courte réelle dans `COLLECTION_PAYLOAD` ;
4. exécuter **Run All Cells** ;
5. surveiller le tableau vivant du lot et des essais ;
6. après une coupure, relancer avec les mêmes paramètres : la campagne active est retrouvée et
   les lots terminés sont ignorés.

À la fin de la durée allouée ou de tous les lots, les cellules suivantes chargent directement les
exports persistants, appliquent la porte scientifique puis entraînent le modèle si le dataset est
identifiable. Le runner autonome en Job reste disponible comme seconde interface et est décrit
dans [`e026-week-unattended.md`](e026-week-unattended.md).

Après une coupure électrique, le notebook interroge aussi PostgreSQL par l'API, réexporte toutes
les campagnes terminales du même hash de payload dans `exports-recovered`, puis déduplique les
reprises par `(payload, prompt, configuration, seed, ECC)`. Une campagne rejouée ne peut donc pas
surpondérer artificiellement sa recette dans le modèle.

L'incident réel et les chiffres de récupération du 17 août 2026 sont consignés dans
[`e026-power-recovery-2026-08-17.md`](e026-power-recovery-2026-08-17.md).

Les scores CLIP/HPS doivent être activés durant la génération. Un recalcul E025 reste possible
pour les campagnes dont les images sont encore présentes.

## Porte d'entraînement

Par défaut, le notebook exige :

- au moins 100 lignes utilisables ;
- au moins 12 textes de prompts distincts ;
- au moins 12 succès et 12 échecs QR-Verify ;
- le texte exact du prompt ;
- la cible QR-Verify.

Ces valeurs autorisent un premier modèle, mais pas une promesse de production. Les abaisser pour
faire disparaître un message d'arrêt produirait seulement des scores artificiels.

## Modèles et validation

Le classifieur et les régressions sont des `ExtraTrees`. Ce choix convient mieux qu'un réseau
neuronal au volume tabulaire actuel et permet d'obtenir une importance des variables ainsi qu'une
dispersion entre arbres.

La validation groupe toutes les lignes partageant le même texte. Un prompt ne peut donc pas être
présent simultanément dans l'entraînement et la validation. Le rapport contient notamment :

- Average Precision QR ;
- ROC AUC QR ;
- Brier avant et après calibration ;
- MAE groupée de chaque cible continue ;
- courbe de calibration ;
- 25 variables les plus importantes ;
- couverture de chaque score.

La dispersion des arbres n'est pas un intervalle statistique garanti. E026 l'utilise seulement
comme pénalité prudente et comme signal d'apprentissage actif.

## Recommandation et apprentissage actif

E026 v1 classe uniquement les configurations déjà observées. Il ne recommande donc pas une
combinaison arbitraire très éloignée du domaine d'entraînement. Le top-K contient :

- la probabilité QR calibrée ;
- son incertitude ;
- sa borne basse ;
- les scores visuels prédits ;
- la saturation et la durée prédites ;
- la configuration JSON exacte.

Le lot actif mélange les trois meilleures recettes et les trois configurations les plus
incertaines. Après génération et QR-Verify, leurs nouvelles lignes rejoignent le dataset. Cette
boucle est plus sûre qu'une exploration exhaustive permanente.

## Inférence réelle E026I

Une recommandation tabulaire ne constitue pas une image. Après l'entraînement, le notebook lance
donc E026I sur dix textes absents du dataset : cinq prompts simples et cinq atypiques. Chaque
prompt reçoit son propre top-3 E026, comparé à `diffqrcoder_stage1` avec trois seeds appariées.
Le budget comparatif est de 120 images. Lorsqu'une recette conseillée est SR-MPGD, le plan v2
ajoute d'abord le Stage 2 SRPG strictement apparié exigé par le laboratoire. Ces prérequis sont
dédupliqués par configuration de Stage 2, ne participent pas au verdict conseiller contre baseline
et peuvent porter le budget GPU total jusqu'à 150 générations pour le plan actuel.

Les pixels restent générés par DiffQRCoder, Cetus-Mix Whalefall et QR Monster v2. L'expression
« généré avec E026 » signifie précisément que la recette a été choisie par le conseiller avant
la génération. Chaque essai conserve le rang, la recette source, les scores prédits et les scores
mesurés. Le verdict rapporte séparément :

- le taux QR-Verify du rang 1 ;
- le taux QR-Verify de toutes les images du top-3 ;
- la couverture prompt/seed lorsqu'au moins une des trois recettes réussit ;
- le taux de la baseline Stage 1 ;
- CLIP-Aesthetic, CLIPScore et HPS v2.1 uniquement parmi les recommandations scannables.

Les taux principaux utilisent toutes les images comparatives planifiées comme dénominateur : une
erreur technique n'est jamais supprimée silencieusement. Les taux suffixés `_generated` décrivent
uniquement les images effectivement produites et servent au diagnostic. Le rapport publie aussi le
nombre d'erreurs et le taux de complétion technique.

Le plan est identifié par le hash du conseiller, des prompts, des recettes, des seeds et du
contexte QR. Son état atomique réside dans `/data/e026-inference/<plan-id>`. Une reprise retrouve
une campagne API déjà créée grâce à son nom déterministe, réexporte un résultat terminal si
nécessaire et ne soumet jamais de nouveau les prompts terminés. Le plan persistant ne contient pas
le payload clair, seulement son SHA-256 et sa longueur.

Dans Jupyter, les images sont visibles sans extraire l'archive sous
`results/<run>/advisor-inference-gallery`. Le sous-dossier `images` contient chaque PNG ; les
fichiers `comparison-seed-*.png` et `measured-winners.png` sont les planches de synthèse.

Les prompts E026I restent un holdout et ne sont pas automatiquement ajoutés aux globes
d'entraînement du même run. Leur réutilisation pour entraîner une version suivante devra être
accompagnée d'un nouveau jeu de prompts d'évaluation réellement inconnus.

### Incident E026I v1 du 17 août 2026

L'archive `20260817T103009Z-e026-prompt-parameter-advisor-v1` contenait bien 120 lignes, mais
seulement 45 images : 30 baselines et 15 SRPG. Les 75 recettes SR-MPGD ont été refusées avant
génération parce que le plan n'avait pas matérialisé leur source SRPG appariée. Les moyennes pandas
ignoraient ensuite les valeurs absentes et annonçaient à tort 100 % pour le top-K. Les chiffres
honnêtes sur le plan v1 sont 15/90 recommandations produites et lisibles, 15/30 couples
prompt/seed couverts, contre 21/30 baselines lisibles. Ces données ne permettent pas de conclure sur
SR-MPGD ni sur la supériorité du conseiller.

Le protocole `e026i-v2-paired-srmpgd` corrige les deux causes : il insère et déduplique les sources
SRPG avant SR-MPGD, puis compte toute mesure manquante comme un échec technique dans le taux
principal. Son changement de protocole produit un nouvel identifiant de plan ; une relance ne peut
donc pas réutiliser l'état v1 incomplet.

## Artefacts

Chaque exécution écrit dans `/data/notebook-runs/<date>-e026-...` :

- `dataset-audit.json` ;
- `policy-dataset-targets.csv` ;
- `training-report.json` ;
- `grouped-validation-predictions.csv` ;
- `validation-and-feature-importance.png` ;
- `feature-importance.csv` ;
- `prooftag-e026-parameter-advisor.joblib` ;
- `recommendations.csv` et `recommendations.json` ;
- `advisor-inference-results.csv` et `advisor-inference-aggregate.csv` ;
- `advisor-inference-evaluation.json` et `advisor-inference-scorecard.png` ;
- `advisor-inference-gallery/` avec les PNG, trois comparaisons appariées et les gagnants ;
- `advisor-inference-audit/` avec le plan, l'état, les prédictions et les exports API ;
- `active-learning-batch.json` ;
- `manifest.json` ;
- une archive `.tar.gz` du run.

## Limites connues

- QR-Verify reste un test logiciel sur l'image, pas une probabilité physique universelle.
- CLIP-Aesthetic, CLIPScore et HPS v2.1 sont des proxys ; les notes humaines doivent être
  conservées.
- Une révision de modèle, de ControlNet, de géométrie ou de pipeline doit être représentée dans
  les paramètres. Un changement non enregistré invalide le modèle.
- Le top-1 prédit ne suffit pas : générer un petit budget top-K, puis appliquer la porte réelle.
- Les performances doivent être publiées séparément sur prompts vus et prompts totalement
  inconnus.
