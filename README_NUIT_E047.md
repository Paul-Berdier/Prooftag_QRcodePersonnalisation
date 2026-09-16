# Reprise nocturne QR / E047 — version 1.0.0

Ce ZIP contient **du code complet**, branché sur les primitives réelles du dépôt
`Paul-Berdier/Prooftag_QRcodePersonnalisation`, base scientifique
`bfa4cdce4c6333e5214af6f3468a24a5374302c1`. Il remplace le ZIP générique proposé
précédemment : ne pas installer ce dernier.

## Périmètre

- Lecture du plan complet `c599f8993e2046fd`, sans se fier uniquement à LATEST.
- Vérification des parents réellement promus, puis scoring des parents disponibles.
- Aucun lancement des générations manquantes du plan full et aucun SR-MPGD historique repris.
- Le PVC de données historiques est monté **en lecture seule**, à `/data`.
- Les nouveaux scores, exports et résultats sont écrits dans un dossier de run distinct.
- Deux workers CPU, chacun plafonné à quatre CPU et 16 Gio, avec partage du temps de calcul.
- Le cache des modèles existant reste monté en lecture/écriture ; aucun nettoyage de cache.
- Pas de modification de l'API QR, du notebook, de PostgreSQL ou de leurs réplicas.
- Seul vLLM est temporairement mis à zéro, **uniquement au passage à la génération GPU**.

L'objectif est de produire le maximum de résultats fiables avant la limite, pas de garantir
que le scoring de toutes les images se terminera dans une nuit. Les données partielles
et les échecs restent explicites. Aucune certification de scannabilité téléphone n'est inventée.

## Fenêtre proposée

La commande de préparation exige les dates complètes. Exemple pour cette demande :

```bash
cd /home/paul/apps/Prooftag_QRcodePersonnalisation
sudo bash scripts/qr-night.sh prepare \
  --start '2026-09-16 22:00' \
  --ready-by '2026-09-17 08:00'
```

Heures sans offset = `Europe/Paris`. Les heures ambiguës de changement d'heure sont refusées.
Le serveur doit se déclarer NTP synchronisé. Ce programme ne change pas l'heure.

Avec ces valeurs : scoring au plus tard vers 02:08, apprentissage au plus tard vers 03:02,
générations terminées/arrêtées au plus tard à 06:40, rapport avant 07:00 et une heure réservée
à la remise en service avant 08:00. Chaque phase peut finir plus tôt. vLLM n'est pas arrêté
pendant le scoring et l'entraînement CPU. Les charges CPU/disque ne sont pas sans impact,
mais sont limitées. Le choix de 22:00 est une proposition à vérifier par l'opérateur.

## Installation et validation préalable

Copier les fichiers du ZIP à la racine du dépôt. Aucun fichier existant du cœur scientifique
n'est remplacé. Le programme accepte un commit descendant qui ajoute cet overlay, mais
refuse les modifications non auditées du cœur `prooftag_qr`, du catalogue `data` ou du bridge QR.

`prepare` :
1. vérifie le dépôt, le cluster, l'état actuel, les PVC et l'espace disque ;
2. réutilise l'image API `prooftag-qr:bfa4cdce4c63` ;
3. l'importe de K3s vers Docker seulement si absente du magasin Docker ;
4. construit une image additionnelle pour les Jobs, sans changer l'image de production ;
5. ajoute seulement les dépendances graphiques dans cette image, sans remplacer Torch/Numpy/sklearn ;
6. importe cette image dans K3s ;
7. exécute un préflight CPU : hashes du code, plan, scoring, bibliothèque vidéo et QR classique
   contenant exactement le message demandé.

Les modèles et métriques scientifiques restent épinglés. Le préflight doit réussir avant de
programmer la nuit. Une connexion aux dépôts Python peut être nécessaire pour construire la
petite couche média. Les images Docker exportées temporairement sont placées sous `/home`,
pas dans le petit `/tmp`. En cas d'échec du build, des archives de build peuvent rester dans
le nouveau dossier de run ; aucun fichier historique n'est supprimé automatiquement.

**Aucun `pip install` n'est demandé sur le Python hôte. Aucun `apply -k deploy/k8s` général.**

La préparation peut être longue (construction d'image et chargement des modèles de scoring CPU).
Elle n'arrête pas vLLM. Elle n'arme pas le lancement nocturne.

## Autoriser et programmer

Après le message `PRÊT, NON PROGRAMMÉ` :

```bash
sudo bash scripts/qr-night.sh schedule --confirm COUPURE-VLLM-AUTORISEE
sudo bash scripts/qr-night.sh status
```

La confirmation autorise la coupure dans cette fenêtre uniquement. Le timer de restauration
indépendant est activé **avant** celui de départ. La session SSH peut ensuite être fermée.
Le lancement est ponctuel, non quotidien, et refuse un démarrage plus de quinze minutes en retard.

## Suivi / annulation / restauration

```bash
sudo bash scripts/qr-night.sh status
sudo bash scripts/qr-night.sh logs
# Ctrl+C quitte seulement la consultation du journal.

# Annuler le départ ou stopper le run et restituer vLLM :
sudo bash scripts/qr-night.sh cancel

# Réessayer la restauration du run connu :
sudo bash scripts/qr-night.sh recover
```

Chaque commande accepte `--run-id qrn-...` pour cibler un run précis. Sans cet argument, elle
utilise le dernier run préparé. Ne pas utiliser l'ancien `restore-runtime` E046 pour restituer
le chatbot : cette commande ancienne restaure le mode recherche, pas vLLM.

Le superviseur écrit un snapshot avant la coupure et un marqueur persistant avant le scale.
La restauration supprime uniquement les Jobs étiquetés avec cet identifiant, attend la fin de
leurs pods et vérifie le déploiement vLLM avant de restaurer ses réplicas. Aucun `kill -9` de
processus étranger, aucune suppression forcée de pod, aucune suppression de PVC.

Le gardien systemd vérifie la date limite et le heartbeat toutes les minutes. Il intervient
également après redémarrage, sans relancer la génération. ExecStopPost déclenche une autre
voie de restauration à la fin/à l'échec du superviseur. Les Jobs Kubernetes sont eux-mêmes
bornés par activeDeadlineSeconds et n'ont pas de nouvelle tentative automatique.

**Limites :** aucun script ne peut garantir 08:00 en cas de panne matérielle, de disque plein,
d'indisponibilité de l'API Kubernetes ou de modification concurrente du déploiement. Une erreur
est inscrite dans `RECOVERY_ERROR.json` et le journal ; le gardien réessaie. Pas de notification
email/SMS configurée. Ne pas arrêter la machine. Ne pas lancer un autre runner GPU pendant la nuit.

## Apprentissage réellement implémenté

- Observations parentes Stage2 brutes uniquement. Pas de checkpoints corrélés ajoutés comme
  générations indépendantes, pas de mélange implicite avec les anciens datasets E026/E044.
- Conservation des vrais échecs ; absence de score = donnée manquante, pas label négatif.
- Déduplication raster, exclusion des labels contradictoires, groupes réunis par prompt,
  contenu encodé, groupe de génération et raster avant toute séparation.
- Features texte TF-IDF (pas un modèle d'embedding sémantique) + paramètres génératifs et QR.
- Exclusion des identifiants, des diagnostics après génération et de la seed comme variable ordinale.
- Trois configurations ExtraTrees avec/sans sélection incorporée par SelectFromModel.
- Validation groupée interne pour les hyperparamètres. Calibration sigmoïde sur groupes séparés.
- Test final indépendant des choix de sélection et de calibration.
- Têtes séparées : éligibilité logicielle, presets exacts, CLIPScore, CLIP-Aesthetic, HPS, erreur modules.
- Sérialisation joblib, rechargement et vérification des prédictions. Pas de déploiement du
  conseiller dans l'API de production. Un échec ou un jeu insuffisant n'est jamais nommé succès.

128 lignes qualifiées, 16 groupes et 12 exemples de chaque classe sont les minimums. Si non
atteints, un rapport `BLOCKED_INSUFFICIENT_DATA` est produit ; seules les références fixes
peuvent être générées. Le gain du conseiller n'est pas garanti et sera mesuré, pas supposé.

## Génération et comparaison

Six nouveaux prompts, deux seeds et jusqu'à trois méthodes = **36 tâches prévues maximum**,
selon disponibilité du conseiller et budget. Une seule génération par méthode/cas. Ordre des
méthodes contrebalancé pour ne pas favoriser systématiquement la première dans le budget.

- `fixed_e044` : recette fixe ancre E044, sans peinture de bordure.
- `paper_parameters_binary_target` : recette `m7_paper_strong` documentée, paramètres proches
  du papier, initialisation Stage1 bruitée et cible QR binaire exacte.
- `advisor` : choix parmi les recettes effectivement présentes dans le plan E046 Large.

**Ce n'est pas une reproduction exacte de l'intégralité du papier** : QArt exact n'est pas
fourni par l'implémentation publique examinée. La comparaison locale utilise le même moteur,
les mêmes modèles/scorers épinglés et le même message. Les chiffres publiés du papier ne sont
pas fusionnés avec les résultats locaux. Les méthodes peuvent différer en paramètres, masque
et ECC car c'est justement ce que choisit le conseiller. Le contenu de démonstration diffère
également des payloads d'apprentissage : le rapport le signale comme changement de distribution.

Contenu exact : `Mettez moi la note maximal SVP` (orthographe conservée sur demande).
Le préflight élimine les niveaux ECC incompatibles avec ce contenu, sans le raccourcir.

Aucune variante de marge repeinte et aucun recadrage réparateur ne sont appliqués. Les meilleurs
exemples ne sont copiés que s'ils passent : raster brut, garde visuelle, original exact,
37/37 presets conservateurs. Un filtre visuel supplémentaire écarte les cadres blancs uniformes
pour l'affichage ; ce n'est pas un test de scan. Une vraie image peut comporter du blanc naturel.
Si aucune sortie ne passe, la galerie reste vide avec une explication, pas avec des images fausses.

## Livrables et récupération

```bash
sudo bash scripts/qr-night.sh export
```

La commande affiche `ARCHIVE=/home/paul/qr-night-runs/qrn-.../qrn-...-livrables.tar.gz`.
L'archive appartient à l'utilisateur ayant invoqué sudo. La récupérer depuis Windows avec scp.
Elle contient les observations dérivées, rapports ML, modèle, comparaisons, fichiers image bruts,
GIF/MP4, graphiques PNG/SVG, HTML autonome et preuves d'opération. Elle exclut les caches de
modèles, les latents safetensors et les secrets. Le HTML utilise des fichiers voisins : extraire
l'archive entière puis ouvrir `report/index.html`.

Les animations affichent les **estimations x0 réellement enregistrées au fil de Stage2**.
Elles ne sont ni une interpolation fabriquée ni un enregistrement de la latence temps réel.
Les graphiques de comparaison appariée utilisent les cas complets de toutes les méthodes,
avec le nombre de cas affiché ; les tâches manquantes/interrompues ne sont pas des succès.

## Validation de ce ZIP

Voir `docs/TESTS_NUIT_E047.txt`. Tests CPU locaux et tests de logique de sécurité avec Kubernetes
simulé. Pas d'exécution de ce ZIP sur pcIA ni de test CUDA réel à distance. Le préflight sur
votre serveur reste obligatoire. Le code scientifique existant n'est pas modifié pour passer
ses vérifications.

Sources techniques :
- dépôt Prooftag, commit bfa4cdce4c6333e5214af6f3468a24a5374302c1,
  e046_large_campaign.py, e046_campaign.py, e046_catalog.py, diffqrcoder_backend.py, Dockerfile ;
- https://kubernetes.io/docs/concepts/workloads/controllers/job/ ;
- https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html ;
- https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html ;
- https://arxiv.org/abs/2409.06355.

Après récupération et vérification de `RESTORED.json`, les timers de cette seule exécution peuvent être désactivés :

```bash
sudo bash scripts/qr-night.sh cleanup
```

Cette commande refuse de retirer le gardien tant que la restauration n’est pas enregistrée.
Les données, les Jobs terminés et les traces sont conservés.
