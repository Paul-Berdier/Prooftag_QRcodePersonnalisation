# E026W — collecte autonome et reprenable

E026W remplit le PVC avec un dataset borné et reprenable pendant l'absence de l'opérateur. Le
runner est un Job Kubernetes CPU : l'API conserve seule la RTX pour DiffQRCoder. Fermer SSH ou le
navigateur n'interrompt pas le Job.

Le même runner est désormais accessible dans le notebook 21. Dans ce mode, le pod Jupyter est
automatiquement basculé en CPU et l'API reste active sur le GPU. Le tableau du notebook montre la
progression, tandis que `state.json`, les CSV et `notebook-progress.jsonl` persistent sur le PVC.
Fermer seulement l'onglet navigateur ne tue pas le kernel. Si le pod ou le kernel disparaît, une
nouvelle exécution reprend le même plan.

## Volume prévu

- 300 prompts déterministes et uniques ;
- quatre familles : simple, scène, détaillé, atypique ;
- 16 recettes observables ;
- trois seeds ;
- 30 campagnes de 480 essais ;
- maximum planifié : **14 400 images** ;
- échéance interne : 162 heures ;
- échéance Kubernetes : 168 heures ;
- arrêt préventif s'il reste moins de 8 Gio sur le PVC de données.

Le nombre réellement produit dépend du temps de Stage 2, de HPS v2.1 et des erreurs éventuelles.
Le runner privilégie la durée disponible : il commence un lot, attend sa fin, exporte le CSV puis
passe au suivant.

## Espace de paramètres

Le plan conserve les profils publics et ajoute des variations contrôlées :

- Stage 1 ;
- SRPG principal et protocole PDF/QArt public ;
- forces Stage 2 0,35, 0,50, 0,65 et 0,80 ;
- poids SRG 250 à 750 ;
- poids perceptuel 1 à 3 ;
- 32, 40 et 50 pas Stage 2 ;
- ControlNet Stage 2 1,10 à 1,50 ;
- SR-MPGD prudent, robuste, gamma 30/100/300, 4/8 itérations et LPIPS 0,10/0,25 ;
- plusieurs configurations Stage 1, CFG et ControlNet.

Le seed est répété mais ne devient pas une grandeur numérique supposée continue dans le modèle
E026.

## Résilience

Le Job écrit après chaque transition dans `/data/e026-week/<plan-id>/state.json`. Chaque campagne
terminée est immédiatement exportée dans le sous-dossier `exports`. Après un redémarrage du Job,
il reprend la campagne active ou saute les lots déjà terminés. Une campagne marquée `interrupted`
peut être relancée jusqu'à trois fois.

Le payload n'est jamais écrit dans l'état ni les manifests publics. Le plan conserve uniquement
son SHA-256 et sa longueur. Kubernetes le fournit au runner par Secret.

Les aperçus et artefacts de debug sont désactivés durant E026W pour ne pas remplir le PVC de 50
Gio. Les images finales, scores, validations et CSV restent persistés. La valeur précédente est
restaurée par la commande `stop`.

## Lancement

Après commit, push et pull, une seule commande mémorise l'état GPU, déploie l'image Git exacte et
lance la collecte :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation

export E026_PAYLOAD_BASE='https://ptag.io/t/w'
./scripts/e026-week-campaign.sh deploy-start
```

Le script :

1. mémorise les réplicas API, notebook et vLLM **avant** toute pause ;
2. crée le Secret Kubernetes ;
3. met vLLM et Jupyter à zéro ;
4. construit, importe et déploie l'image portant le hash Git exact ;
5. vérifie le payload, les 14 400 essais et les scores CLIP/HPS avant le lancement ;
6. désactive les artefacts de debug ;
7. lance le Job CPU avec la même image ;
8. laisse l'API produire les campagnes une par une.

La commande `start` reste disponible lorsque l'image applicative correcte a déjà été déployée.

Le payload doit être court et réel. Il est identique sur ce premier plan afin de ne pas introduire
un changement de géométrie non maîtrisé pendant l'absence.

## Suivi

```bash
./scripts/e026-week-campaign.sh status
```

ou en continu :

```bash
./scripts/e026-week-campaign.sh logs
```

La fermeture du terminal de suivi n'arrête rien.

## Reprise après incident

```bash
export E026_PAYLOAD_BASE='https://ptag.io/t/w'
./scripts/e026-week-campaign.sh resume
```

Cette commande recrée seulement le Job ; le fichier d'état et les exports restent sur le PVC.
Utiliser exactement le même payload pour retrouver le même identifiant de plan.

## Arrêt et restauration

```bash
./scripts/e026-week-campaign.sh stop
```

La suppression du Job envoie `SIGTERM`. Le runner demande l'annulation de la campagne courante,
puis le script restaure la charge GPU précédente et supprime le Secret. Les données E026 restent
sur le PVC.

À l'échéance de sept jours, le Job s'arrête mais l'API reste disponible et vLLM reste volontairement
en pause. Exécuter `stop` au retour pour restaurer l'état antérieur.

## Entraîner le conseiller au retour

Les CSV sont directement visibles par le notebook 21 via
`/data/e026-week/*/exports/*.csv`. Après `stop`, rouvrir le notebook E026 et exécuter **Run All** :
aucune copie intermédiaire vers `/workspace/imports` n'est nécessaire.
