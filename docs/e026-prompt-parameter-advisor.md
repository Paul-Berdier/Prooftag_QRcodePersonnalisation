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

## Collecte E026A

Le notebook contient un plan initial de 24 prompts : six simples, six scènes, six détaillés et
six atypiques. Il retient dix recettes publiques/pinnées et deux seeds. La campagne totale compte
480 essais, découpés en quatre lots de 120 afin qu'une interruption ne perde pas tout le travail.

Ce premier plan est un **minimum pilote**, pas une preuve de généralisation industrielle. Après
le pilote, la cible recommandée reste 5 000 à 15 000 sorties réparties sur au moins 100 prompts.

Pour profiter d'une semaine de calcul sans garder Jupyter connecté, utiliser la collecte E026W
décrite dans [`e026-week-unattended.md`](e026-week-unattended.md). Elle étend ce pilote à 300
prompts, 16 recettes, trois seeds et un maximum de 14 400 essais persistants.

Pour écrire les quatre manifests :

1. ouvrir `21_e026_prompt_parameter_advisor.ipynb` ;
2. définir `COLLECTION_PAYLOAD` ;
3. exécuter les cellules 1 à 3 ;
4. récupérer les JSON dans `collection-plan` ;
5. arrêter Jupyter pour rendre la RTX à l'API ;
6. soumettre les campagnes l'une après l'autre au laboratoire ;
7. exporter chaque résultat en CSV.

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
