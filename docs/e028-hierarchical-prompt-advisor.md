# E028 — conseiller hiérarchique DiffQRCoder

## Décision corrigée après E027

E027 a mesuré 300 triplets complets. Forcer SR-MPGD n'a pas amélioré la porte logicielle :
`242/300` contre `244/300` pour la meilleure sélection mesurée. Mais sa politique de cascade
autorisait encore Stage 1 à la livraison. Cette règle est invalide pour Prooftag : les essais
téléphone montrent que Stage 1 reste une référence esthétique et non une sortie exploitable.

E028 impose donc les règles suivantes :

- Stage 1 sert uniquement de source pour Stage 2 ; il n'est jamais livrable ;
- le conseiller choisit des paramètres distincts pour chaque prompt et chaque étage ;
- Stage 2 SRPG est le premier candidat de production ;
- SR-MPGD est un repli si Stage 2 échoue au payload exact ou à la tolérance QR-Verify `0,80` ;
- la comparaison esthétique n'intervient qu'après la porte QR et la garde de saturation ;
- QR-Verify reste un test logiciel sur fichier, pas une garantie de lecture physique.

## Chaîne expérimentale

Pour chacun des trente prompts inconnus et chacune des trois seeds, le notebook construit :

```text
chaîne fixe : Stage 1 ─► Stage 2 ─► SR-MPGD

conseiller : 2 Stage 1
               └─► 2 Stage 2 par Stage 1
                       └─► 1 SR-MPGD exact par Stage 2
```

Cela donne treize états par prompt/seed et `30 × 3 × 13 = 1 170` images. Tous les états sont
calculés pendant l'expérience afin d'obtenir les contrefactuels nécessaires à l'apprentissage.
La politique de production rejouée dans le rapport ne compte cependant SR-MPGD que lorsque son
Stage 2 parent échoue.

## Où intervient le modèle

Le modèle `E026ParameterAdvisor` reçoit l'embedding du prompt, la géométrie QR, la longueur du
payload et les paramètres de chaque recette candidate.

1. **Conseiller Stage 1 :** sélection d'un profil structurel et d'un profil esthétique parmi les
   variantes de pas, CFG et force ControlNet.
2. **Conseiller Stage 2 :** chaque candidat contient les paramètres du Stage 1 choisi et ses
   propres paramètres SRPG. Le modèle sélectionne un profil robuste et un profil esthétique.
3. **Conseiller SR-MPGD :** chaque candidat conserve exactement la configuration Stage 2 et ne
   varie que les paramètres SR-MPGD. Le modèle choisit le raffinement lié à ce latent.

Le premier modèle prend ses décisions avant la génération. Après E028, deux nouveaux datasets
conditionnels ajoutent les mesures réelles du parent : QR exact, tolérance, HPS, CLIP, saturation
et MER. Ils permettent d'entraîner un conseiller Stage 2 conditionné par son Stage 1 et un
conseiller SR-MPGD conditionné par son Stage 2.

Limite honnête du premier passage : les anciens exports n'associent pas encore chaque Stage 1 à
la réussite aval de plusieurs Stage 2. Le premier choix Stage 1 optimise donc ses propres proxys
structurels et esthétiques, ainsi que la configuration aval prédite. Les chaînes E028 servent
précisément à créer la cible manquante « utilité réelle de ce Stage 1 pour la suite ».

Les variantes Stage 1 ajoutées pour élargir le domaine sont des points expérimentaux lorsque leur
compteur d'observations vaut zéro. Le notebook affiche ce compteur : une prédiction sur un tel
point est une proposition à mesurer, jamais une preuve qu'il est optimal.

## Appariement prouvé

Les noms de recettes ne suffisent pas. `e028-pairing-audit.csv` vérifie :

- le `run_id` de la source Stage 1 réutilisée ;
- le hash de l'image Stage 1 lorsque deux recettes mathématiquement identiques partagent le cache ;
- le `run_id` de la source Stage 2 ;
- le SHA-256 du latent Stage 2 ;
- `stage2_pairing_status=exact_reuse` ;
- `diffqrcoder_stage2_pairing_exact=1` émis par le backend.

Une sortie générée sans ces preuves arrête le notebook. Les erreurs techniques restent visibles,
mais ne sont jamais converties en succès.

## Politiques comparées

- `fixed_cascade` : chaîne fixe, Stage 2 puis SR-MPGD si nécessaire ;
- `advisor_top1` : meilleure chaîne prédite, même cascade ;
- `advisor_best_of_chains` : plusieurs chaînes conseillées, sélection QR-first des résultats.

L'ordre final est : payload exact, tolérance QR-Verify, absence de saturation, HPS v2.1,
CLIP-Aesthetic, CLIPScore. Une image sous la porte QR peut être conservée comme donnée
expérimentale, mais jamais déclarée livrable.

## Fichiers produits

- `e028-state-results.csv` : prédictions et mesures de tous les états ;
- `e028-pairing-audit.csv` : preuve de réutilisation exacte ;
- `e028-policy-decisions.csv` et `e028-policy-report.json` : décisions par contexte ;
- `e028-policy-scorecard.png` : QR, esthétique et coût ;
- `e028-gallery/` : toutes les images et planches appariées ;
- `e028-stage2-conditional-dataset.jsonl` ;
- `e028-srmpgd-conditional-dataset.jsonl` ;
- les modèles conditionnels seulement si leur porte de données est satisfaite ;
- `manifest.json` : protocole, limites et empreinte du conseiller.

## Déploiement et lancement

Sur le serveur Linux, depuis un dépôt propre et à jour :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
bash scripts/deploy-e028-notebook.sh
```

Puis sur le PC Windows :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\notebook-remote.ps1 -Notebook 23_e028_hierarchical_prompt_advisor.ipynb
```

Le script de déploiement construit et déploie l'API et le notebook depuis le même commit avant de
démarrer Jupyter. Il refuse un dépôt sale et contrôle les deux tags immuables après le rollout.
