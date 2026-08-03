# Laboratoire Web DiffQRCoder

## Périmètre de la reprise

Depuis le 29 juillet 2026, le laboratoire ne compare plus les anciennes
réparations Prooftag, FreeQR, Nacholmo ou les rediffusions locales. Une campagne
contient uniquement :

1. le QR binaire témoin ;
2. le Stage 1 public de DiffQRCoder ;
3. le Stage 2 SRPG binaire initialisé selon l’équation 9 du papier ;
4. la sélection automatique du meilleur Stage 1 / Stage 2 ;
5. facultativement, le Stage 2 suivi du SR-MPGD des équations 13–14 ;
6. des ablations désactivées par défaut (force de bruit et QArt public).

Le socle est figé au commit DiffQRCoder
`e24ea73ee2e13c7e6e87cb422e8b11784e70ae00`, avec Cetus-Mix Whalefall et
`monster-labs/control_v1p_sd15_qrcode_monster`, sous-dossier `v2`.

```mermaid
flowchart LR
    Q["QR v3 / M / masque 4"] --> S1["Stage 1\nCetus-Mix + QR Monster v2"]
    S1 --> N["VAE + bruit\nEq. 9"]
    Q --> TARGET["Cible Stage 2\nQR binaire exact"]
    N --> S2["Stage 2\nDDIM + SRPG"]
    TARGET --> S2
    S2 --> M["SR-MPGD Eq. 13–14\noptionnel"]
    Q --> R["Capacité du QR témoin\npar décodeur/scénario"]
    R --> V["Validation automatique normalisée"]
    S1 --> V
    S2 --> V
    M --> V
    V --> A["SSR normalisé, payload, MER exact,\nCLIP-aesthetic, CLIPScore"]
    A --> H["Verdict humain\nesthétique + scan téléphone"]
```

## Ce qui vient directement du dépôt public

- classe `DiffQRCoderPipeline` chargée depuis le dépôt épinglé ;
- scheduler DDIM ;
- Stage 1 ControlNet text-to-image ;
- Stage 2 avec `ScanningRobustLoss` et perte perceptuelle VGG ;
- QR Monster v2 et Cetus-Mix Whalefall ;
- SRG, PG, ControlNet, CFG, pas et ETA configurables ;
- QR v3, masque 4, modules de 20 px et source 740×740 ;
- prétraitement en 736×736 et crop de 78 px, laissant un cœur de
  580×580, soit 29×29 modules de 20 px.

Le laboratoire ne recouvre jamais une sortie artistique avec le QR binaire.
Une image non scannable reste rejetée et visible.

## Reconstruction conforme à l’algorithme du papier

Le dépôt public ne reproduit pas seul tout l’algorithme décrit dans le papier :

- `PerceptualLoss.forward` reconstruit un tenseur avec `torch.tensor([...])`,
  ce qui détache les pertes VGG du graphe. Le wrapper emploie `torch.stack`,
  sans changer la formule ;
- `_run_stage2` génère normalement un nouveau bruit aléatoire. Le wrapper encode
  réellement l’image du Stage 1 avec le VAE puis lui ajoute le bruit DDIM au
  timestep de départ, conformément à l’équation 9 ;
- le constructeur Reed–Solomon de la cible `Qart(x̂, y)` n’est pas publié. Le
  wrapper utilise donc le QR binaire exact comme cible de repli : il est moins
  proche du Stage 1, mais son payload et sa matrice sont garantis ;
- le SR-MPGD public réutilise la loss pondérée du Stage 2. Le wrapper applique
  séparément l’équation 13, `LSR + 0,01 × LPIPS`, puis l’équation 14 avec
  `gamma=1000`.

SR-MPGD décode et valide chaque état, s’arrête en cas de succès strict ou de
gradient non fini et livre le meilleur état observé — jamais aveuglément la
dernière itération. Ces corrections sont déclarées dans `/v1/lab/schema`. Le
commit amont n’est pas modifié sur disque.

## Limite honnête de la cible QArt

Le papier transforme le QR avec QArt afin de rapprocher son motif de l’image
Stage 1. Le dépôt officiel consomme cette image, mais ne contient pas le
constructeur. Une ancienne version Prooftag avait fabriqué un hybride coloré
en remplaçant des centres de modules : ce n’était pas QArt et ce raster était
ensuite binarisé par la loss SRL comme s’il s’agissait d’un QR valide.

Cette imitation est supprimée. Le laboratoire passe désormais le QR binaire
exact à ControlNet et à SRL. Un vrai QArt ne pourra revenir que dans un mode
expérimental séparé, avec preuve de décodage du payload Prooftag avant toute
diffusion.

Pour comparer des paramètres, le Stage 2 reçoit une seed dérivée explicite
(`seed + srpg_seed_offset`). Les recettes d’un même prompt et d’une même seed
peuvent ainsi partager le même bruit Stage 2.

## Utilisation

Depuis Windows :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\lab-remote.ps1
```

L’interface s’ouvre sur `http://127.0.0.1:18080/lab`.

Pour créer une campagne :

1. saisir une URL Prooftag courte compatible avec QR v3 ;
2. ajouter une ligne par prompt au format
   `identifiant | prompt | negative prompt` ;
3. ajouter une ou plusieurs seeds séparées par des virgules ;
4. activer les sorties à comparer ;
5. dupliquer une recette pour tester d’autres paramètres ;
6. lancer la campagne.

Chaque combinaison `prompt × seed × recette` devient un résultat indépendant.
La limite est de 500 résultats. Les générations sont séquentielles pour ne pas
charger plusieurs pipelines en VRAM.

### Paramètres exposés

- Stage 1 : pas, CFG, poids ControlNet ;
- Stage 2 : initialisation papier ou bruit public, quantité de bruit, pas,
  poids ControlNet, SRG `λ1`, PG `λ2`, ETA et seed offset ;
- cible Stage 2 : QR binaire exact, non modifiable dans le profil de production ;
- artefacts : fréquence des aperçus intermédiaires ;
- SR-MPGD : itérations, `gamma`, poids LPIPS et MER initial maximal ;
- avancé : modèles, commit et géométrie QR.

Les valeurs initiales sont 40 pas, CFG 7,5, ControlNet 1,35, SRG 500, PG 2
et ETA 0. SR-MPGD démarre à 20 itérations, `gamma=1000` et
`LPIPS=0,01`, valeurs annoncées par le papier. Ce sont des points de
départ, pas une garantie de lecture.

## Scores automatiques par image

Chaque image finale est testée avec les décodeurs installés et les scénarios de
dégradation du projet. L’interface affiche :

- **SSR normalisé** : proportion des cas relisant le payload parmi ceux que le
  QR témoin sait lui-même passer ;
- **SSR brut et capacité du témoin** : valeurs d'audit non masquées ;
- **Payload original** : réussite sans dégradation ;
- **MER** : taux de modules incorrects ;
- **CLIP-aesthetic**, **CLIPScore** et similarité CLIP ;
- temps de génération, validation et total ;
- initialisation réellement employée, pas effectifs et erreur des centres QArt ;
- variation par rapport au Stage 1 et détection de divergence/saturation ;
- paramètres SR-MPGD réellement appliqués et itération retenue ;
- tableau `décodeur × scénario × résultat × latence`.

Le SSR de décision est normalisé par le QR témoin. Un couple
`décodeur × dégradation` que le QR binaire propre ne sait pas relire est exclu
de la porte de livraison, mais reste conservé dans le SSR brut. Cela corrige le
cas observé où le témoin obtenait 38/39 et était lui-même rejeté par une porte
39/39 impossible.

Le MER Stage 2/SR-MPGD est mesuré sur la géométrie réellement optimisée par
DiffQRCoder : crop 78 px, cœur 580×580 et modules entiers de 20 px. Il n'est
plus calculé en découpant naïvement le canvas 736 px en 37 cellules.

Une sortie est `accepted` seulement si tous les décodeurs originaux supportés
par le témoin relisent le payload et si le SSR normalisé atteint le seuil
configuré, actuellement 100 %. Une alerte couleur demande une inspection
humaine ; seule une divergence forte rejette automatiquement l'image.

### Recette par défaut après la campagne fc403349

- cible Stage 2 : QR binaire exact ;
- force de bruit : `0,65`, pour préserver davantage le Stage 1 que la force
  `1,0`, qui démarrait presque dans du bruit pur ;
- sélection automatique : mode de livraison optionnel qui conserve d'abord une
  sortie acceptée, puis la moins altérée. Il est désactivé dans les campagnes
  visuelles pour ne pas afficher une copie supplémentaire de la sortie retenue ;
- SR-MPGD : désactivé par défaut et lancé seulement si le MER Stage 2 est au
  plus `12 %` ;
- QArt public : expérimental et désactivé, car les quatre sorties testées ont
  échoué au téléphone malgré une cible valide avant diffusion ;
- ablation de force disponible : `0,35 / 0,50 / 0,65 / 0,80`.

## Validation humaine à la chaîne

Cliquer une vignette, puis enregistrer :

- esthétique bonne ou mauvaise ;
- scannable, non scannable ou non testé sur téléphone ;
- note esthétique de 1 à 10 ;
- notes libres.

Le bouton **Enregistrer et suivant** ouvre immédiatement l’image suivante. La
campagne agrège le taux automatique strict, SSR, CLIP-aesthetic, CLIPScore,
scans humains positifs, esthétiques positives et nombre d’images évaluées.
L’export CSV contient les paramètres, métriques et verdicts humains.

## Déploiement sans ambiguïté de version

Après commit et push, sur le serveur :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git pull --ff-only origin main
bash scripts/deploy-app-image.sh
```

Le script refuse un dépôt sale, construit une image taguée avec le commit Git,
vérifie le commit DiffQRCoder, importe l’image dans containerd K3s, applique la
migration `0005_e017_phone_calibration`, attend le rollout puis contrôle dans le pod les
profils de production et d'ablation, l’import DiffQRCoder et la version
`20260803-e017-phone-proxy-1` des assets Web, l’initialisation Stage 1 bruitée,
la cible binaire exacte, la force `0,65`, le repli automatique et le seuil
SR-MPGD.

Le taux publiable sera calculé sur les sorties artistiques réellement générées,
jamais sur le QR témoin ni sur une réparation cachée.

La phase E017 est décrite dans
[`e017-phone-proxy-calibration.md`](e017-phone-proxy-calibration.md). Le score
Phone Proxy affiché dans les cartes est une métrique de calibration et non une
porte d'acceptation. Les évaluations téléphone doivent maintenant enregistrer
les essais, les réussites et l'appareil pour rendre la comparaison statistique
exploitable.
