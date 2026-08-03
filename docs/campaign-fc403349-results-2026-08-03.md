# Audit de la campagne fc403349 et décision de reprise

Source analysée : `prooftag-lab-fc403349-31ce-4075-bf52-4360082424a0.csv`,
20 lignes, quatre prompts et cinq sorties par prompt.

## Résultats observés

| Sortie | Scan téléphone | Esthétique humaine | SSR automatique moyen | MER moyen |
|---|---:|---:|---:|---:|
| QR témoin | 4/4 | contrôle | 97,4 % | 0,0 % |
| Stage 1 | 1/4 | 4/4 | 0,6 % | 15,0 % |
| Stage 2 SRPG binaire | 3/4 | 2/4 | 24,4 % | 9,9 % |
| Stage 2 SRPG QArt | 0/4 | 2/4 | 1,9 % | 38,9 % |
| QArt puis SR-MPGD | 0/4 | non concluant | 2,6 % | 38,4 % |

La valeur 97,4 % du QR témoin correspond à 38 validations réussies sur 39.
Une porte absolue à 39/39 rendait donc mathématiquement impossible
l'acceptation de toute image, y compris du contrôle propre.

## Ce qui fonctionnait réellement

- Le Stage 1 était réutilisé entre les méthodes.
- Le latent propre du Stage 2 était réutilisé avant SR-MPGD ; le temps moyen
  tombait d'environ 47,1 s pour SRPG à 20,1 s pour SR-MPGD.
- La cible QArt expérimentale était décodable avant diffusion.
- Le Stage 2 binaire améliorait nettement la lecture physique : 3/4, contre
  1/4 au Stage 1.

## Causes des résultats incohérents

1. **Porte impossible.** Le score candidat était comparé à 100 % de tous les
   tests, alors que le témoin n'en passait que 97,4 %.
2. **Deux géométries MER.** Le service divisait parfois le canvas 736 px en
   37 cellules égales, tandis que SR-MPGD évaluait le cœur 580 px en modules
   de 20 px. Les valeurs ne mesuraient pas la même chose.
3. **Bruit Stage 2 maximal.** À force `1,0`, le coefficient conservant le
   Stage 1 était proche de `0,078` et celui du bruit de `0,997` : ce n'était
   presque plus une rediffusion du Stage 1.
4. **QArt non généralisant.** Malgré une cible pré-diffusion valide, la
   diffusion a produit 0/4 scans téléphone. SR-MPGD ne peut pas sauver de
   façon fiable une sortie initiale aussi éloignée.
5. **Garde couleur trop binaire.** Une alerte modérée pouvait écarter une
   image lisible, tandis qu'il fallait distinguer une dérive à inspecter d'une
   saturation réellement destructrice.

## Corrections décidées

- SSR normalisé par la capacité du QR témoin, avec conservation des scores
  brut et témoin dans chaque résultat ;
- MER unique sur la géométrie officielle DiffQRCoder ;
- cible binaire comme voie principale, QArt isolé comme ablation ;
- force de bruit par défaut `0,65` et sweep apparié `0,35/0,50/0,65/0,80` ;
- SR-MPGD seulement si le MER initial est inférieur ou égal à `12 %` ;
- garde couleur en deux niveaux : avertissement humain puis rejet fort ;
- mode automatique qui compare Stage 1 et Stage 2 et préserve le Stage 1 si le
  Stage 2 perd la lecture ou diverge ; ce mode de livraison reste désactivé
  pendant les campagnes visuelles pour éviter une vignette dupliquée ;
- réutilisation des Stage 1 et Stage 2 appariés, sans régénération cachée.

## Ce que cette correction ne prétend pas

Elle ne prouve pas encore 99 % de lecture physique. Elle rend enfin la prochaine
campagne mesurable et comparable. Le taux publiable devra être calculé sur un
lot de prompts non vus, plusieurs seeds, plusieurs téléphones et des impressions,
en comptant toutes les générations artistiques et sans inclure le QR témoin.
