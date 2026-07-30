# DiffQRCoder : cible QArt, saturation et chaînage des étapes

Date de décision : 2026-07-30.

## Problème observé

Certaines sorties du Stage 2 présentaient une dominante jaune/cyan, des
couleurs écrêtées et une perte de lisibilité. Deux problèmes différents
étaient confondus :

1. le Stage 2 recevait un QR binaire exact alors que l'article décrit une
   cible QArt rapprochée visuellement du Stage 1 ;
2. le laboratoire lançait chaque méthode comme une génération indépendante.
   Le profil SR-MPGD recalculait donc le Stage 2 au lieu de poursuivre depuis
   le latent final du profil SRPG correspondant.

Le `ScanningRobustLoss` officiel travaille sur la luminance et les centres de
modules. Il ne pénalise ni la saturation chromatique ni l'écrêtage RGB. Une
image peut donc améliorer sa perte QR tout en devenant visuellement saturée.

## Pipeline retenu

```text
QR de référence
      |
      v
Stage 1 ControlNet (une seule fois par prompt/seed/configuration)
      |
      +--> cible QArt construite depuis le Stage 1
      |    - plusieurs seuils testés
      |    - validation par tous les décodeurs sur l'image originale
      |    - meilleur candidat strict retenu
      |
      v
Stage 2 SRPG (une seule diffusion)
      |
      +--> image finale SRPG
      +--> latent final z0 conservé en RAM CPU
      |
      v
SR-MPGD
      - reprend exactement le latent z0 du Stage 2
      - aucune nouvelle diffusion Stage 1 ou Stage 2
      - optimise SRL + lambda LPIPS
```

Le cache Stage 2 est apparié par prompt, prompt négatif, seed, payload,
géométrie QR, modèle et tous les paramètres qui influencent Stage 1/Stage 2.
Les paramètres propres à SR-MPGD ne font volontairement pas partie de cette
clé.

## Contrat de payload QArt

L'implémentation publique QArt ajoute un fragment à l'URL pour exploiter les
degrés de liberté du codage Reed-Solomon. Le résultat n'est donc pas identique
octet pour octet au payload initial.

Le contrat accepté est :

- même schéma, hôte, chemin et requête ;
- fragment ignoré ;
- le fragment n'est jamais envoyé au serveur HTTP par le navigateur.

L'interface affiche ce contrat comme `URL canonique`, jamais comme
`payload exact`. Le mode `binary_exact` reste disponible comme témoin
d'ablation.

Si aucun candidat QArt n'est lu par tous les décodeurs sur l'image originale,
la génération s'arrête en erreur. Une cible QArt invalide n'est jamais donnée
au Stage 2.

## Garde-fous contre la saturation

Une sortie SRPG/SR-MPGD est rejetée, même si elle est scannable, lorsqu'elle
dépasse un des seuils configurés :

- ratio de pixels modifiés ;
- changement absolu moyen par rapport au Stage 1 ;
- ratio de pixels écrêtés ;
- ratio de canaux RGB écrêtés ;
- hausse de saturation moyenne ;
- hausse du ratio de pixels très saturés.

Les mesures et la cause exacte du rejet sont enregistrées dans les
diagnostics du run et affichées dans le laboratoire.

## Aperçus de diffusion

Les anciens aperçus pouvaient montrer le latent bruité `z_t`, ce qui donnait
l'impression d'une image volontairement détruite. Les nouveaux aperçus
montrent l'estimation propre `x0` calculée par la boucle officielle, sans
modifier le latent ni le résultat final.

## Validation requise après déploiement

Pour chaque prompt/seed :

1. `diffqrcoder_stage1` doit produire une seule image source ;
2. `diffqrcoder_srpg` doit afficher `Stage 2 réutilisé : non` ;
3. `diffqrcoder_srmpgd` doit afficher `Stage 2 réutilisé : oui` ;
4. la cible doit afficher `QArt réel / URL canonique` ;
5. le SSR robuste, les lectures originales, le MER, CLIPScore,
   CLIP-Aesthetic et les métriques de saturation doivent être présents ;
6. le test manuel doit noter séparément esthétique et lecture téléphone.

Cette correction supprime les régénérations intermédiaires et empêche la
livraison d'une sortie saturée. Elle ne garantit pas qu'une configuration
SRPG donnée atteigne 100 % de lecture : le laboratoire sert précisément à
mesurer ce taux sur plusieurs prompts et seeds.
