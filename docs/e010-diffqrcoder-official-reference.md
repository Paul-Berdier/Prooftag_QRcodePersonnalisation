# E010 — baseline DiffQRCoder officielle et observable

## Décision

E010 revient à l'algorithme public DiffQRCoder comme baseline principale. Les essais Nacholmo
restent archivés, mais leur esthétique quadrillée ne justifie pas de poursuivre leur optimisation
avant d'avoir reproduit correctement le modèle à l'origine du projet.

Cette expérience n'utilise pas le backend SRPG Prooftag existant. Ce backend est un img2img
personnalisé, tandis que le code public DiffQRCoder réalise deux débruitages ControlNet et injecte
la Scanning Robust Perceptual Guidance pendant chaque pas du second. Comparer leurs images sans
séparer les implémentations était méthodologiquement faux.

## Sources et version figée

- dépôt : `jwliao1209/DiffQRCoder` ;
- commit exécuté : `e24ea73ee2e13c7e6e87cb422e8b11784e70ae00` ;
- pile publiée : PyTorch 2.6.0, Diffusers 0.32.2, Transformers 4.48.3 et Torchvision 0.21.0 ;
- fondation : Cetus-Mix Whalefall fp16 ;
- ControlNet : QR Code Monster, sous-dossier `v2`.

Le commit et les versions sont contrôlés pendant le build et répétés dans chaque `manifest.json`.
L'image Jupyter est volontairement indépendante de l'image API afin de ne pas modifier la
production avant la fin de l'expérience.

## Protocole pas à pas

1. Construire un QR version 3, correction M, masque 4, quiet zone de quatre modules et modules de
   20 pixels. `fit=False` interdit tout changement implicite de version.
2. Générer le Stage 1 en text2img ControlNet avec 40 pas DDIM, CFG 7,5 et ControlNet 1,35.
3. Sauvegarder l'état du générateur CUDA juste après le Stage 1.
4. Exécuter un Stage 2 SRPG avec SRG 500 et PG 3.
5. Rejouer le Stage 2 depuis le même état aléatoire et ajouter 20 itérations SR-MPGD à 0,1.
6. Capturer l'estimation décodée et l'erreur de modules tous les cinq pas.
7. Tester chaque sortie avec tous les décodeurs disponibles et treize scénarios de dégradation.
8. Mesurer MER, CLIP-aesthetic, CLIPScore, durée, changement perceptuel et métriques d'image.
9. Autoriser une livraison uniquement à 100 % des validations exactes. L'esthétique ne départage
   que les candidates ayant déjà franchi cette porte.
10. Exporter images, traces, validations, CSV, graphique final, gabarit physique et archive tar.gz.

Les variantes Stage 2 sont appariées. Une différence entre elles ne peut donc pas être attribuée à
un nouveau bruit initial. Le Stage 1 reste présent dans le tableau pour mesurer le gain réel du
raffinement.

## Paramètres publiés et paramètres exploratoires

Le papier indique QR version 3/M/masque 4, modules 20 px, 40 pas par stage, ControlNet 1,35 et une
ablation favorable autour de SRG 500 / PG 3. Le nombre d'itérations SR-MPGD n'est pas suffisamment
spécifié pour être présenté comme une reproduction exacte ; les 20 itérations du notebook sont
donc explicitement étiquetées comme valeur exploratoire du code public. Elles devront être
balayées après la première exécution complète.

Le dépôt public et le pseudo-code du papier présentent aussi des écarts, notamment sur
l'initialisation du second stage et le détail de la projection finale. E010 exécute le dépôt tel
qu'il est au commit figé et consigne cet écart au lieu de réécrire silencieusement l'algorithme.

Un défaut bloquant du dépôt est corrigé explicitement dans le notebook : `PerceptualLoss` agrège
les pertes VGG avec `torch.tensor`, ce qui détache le graphe autograd et peut replacer la valeur sur
CPU. E010 remplace uniquement cette agrégation par `torch.stack(losses).mean()`. La formule, les
poids et les échelles ne changent pas. Ce correctif est visible dans la cellule d'initialisation,
répété dans le manifest et exporté dans `upstream-patches.json` ; il n'est donc jamais présenté
comme le commit amont intact.

## Critères de lecture des résultats

- `26/26` est une porte logicielle locale si OpenCV et ZBar sont disponibles ; le total réel est
  inscrit dans le manifest.
- Un résultat sur le payload témoin ne prouve rien pour une URL longue. Le premier essai Prooftag
  doit conserver la version 3 grâce à un service d'URL courte.
- Une candidate non stricte est nommée `NOT_DELIVERABLE`, même si elle est visuellement réussie.
- La validation physique reste obligatoire sur plusieurs téléphones, distances, angles,
  éclairages, écrans et impressions.
- Les 99–100 % publiés sont des résultats de l'article sur son jeu de 100 prompts, pas un taux
  Prooftag acquis.

## Exécution

Sur le serveur, après récupération du commit du projet :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
docker build -t prooftag-qr:dev .
docker build -f Dockerfile.notebook -t prooftag-qr-notebook:dev .
docker save prooftag-qr:dev prooftag-qr-notebook:dev | sudo k3s ctr images import -
kubectl apply -k deploy/k8s
```

Depuis PowerShell sur le PC :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\notebook-remote.ps1 -Notebook 07_diffqrcoder_official_live.ipynb
```

Dans Jupyter, lancer `Run > Run All Cells`. À la fin, récupérer l'archive avec la commande `scp`
imprimée par la dernière cellule, puis restaurer le GPU :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```

## Suite conditionnelle

Après au moins une exécution complète, comparer d'abord Stage 1, SRPG et SRPG + SR-MPGD. Si aucune
candidate n'est stricte, balayer SRG, PG, nombre d'itérations et taux SR-MPGD sur plusieurs prompts
et seeds en gardant les essais appariés. Si la scannabilité est stable mais l'esthétique insuffisante,
alors seulement ajouter un LoRA Prooftag ou fine-tuner le ControlNet. Le fine-tuning ne doit pas
servir à masquer une reproduction incorrecte de la baseline.
