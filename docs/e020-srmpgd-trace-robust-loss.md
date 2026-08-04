# E020 — trajectoire SR-MPGD et loss de lecture robuste

Date : 4 août 2026.

## Décision issue d’E019B

La sonde `d6c43e99` a exécuté cinq gamma appariés (`10`, `30`, `100`, `300`,
`1000`) sur le même latent SRPG. Les cinq sorties ont retenu l’itération zéro.
Le gamma était bien appliqué et les valeurs 300/1000 atteignaient la borne de
pas RMS, mais aucun état ne dépassait le SSR initial de 2,56 %. Le MER initial
était déjà nul alors que les trois décodeurs échouaient sur l’image originale.

La recherche d’hyperparamètres est donc suspendue : la loss module-centre ne
représente pas l’échec réel du scanner.

## Instrumentation obligatoire

Chaque exécution SR-MPGD écrit désormais `srmpgd_trace.json`. Pour l’état zéro
et chaque mise à jour, il contient :

- SSR, MER, SRL officielle et objectif total ;
- composantes flou, réduction, luminosité et contraste ;
- LPIPS, déplacement latent, taille du pas ;
- garde esthétique, gain QR et éligibilité ;
- état finalement retenu et cause d’arrêt.

Le Web Lab affiche cette trajectoire sous les images. Les métriques agrégées du
meilleur état tenté sont également exportées dans le CSV, même si l’état zéro
reste la sortie sûre.

## Ablation robuste

Le profil `diffqrcoder_srmpgd` reste le témoin basé sur la loss publique
DiffQRCoder. Le nouveau profil `diffqrcoder_srmpgd_robust` utilise exactement le
même latent Stage 2, les mêmes quatre itérations, gamma 100, LPIPS 0,10 et les
mêmes gardes E019. Une seule variable change : la loss SRL est moyennée sur :

1. l’image originale ;
2. un flou moyen 3 × 3 ;
3. une réduction bilinéaire à 75 %, puis retour à la taille initiale ;
4. deux expositions à 80 % et 120 % ;
5. un contraste réduit à 75 %.

Les poids initiaux sont respectivement `1 / 1 / 0,5 / 1` pour flou,
réduction, luminosité et contraste. Cette extension ne remplace pas la loss
publique : elle constitue une ablation appariée et désactivable paramètre par
paramètre dans le Web Lab.

## Protocole suivant

Lancer un seul prompt avec cinq méthodes actives : QR témoin, Stage 1, SRPG,
SR-MPGD officiel et SR-MPGD robuste. Comparer d’abord les trajectoires. Une
campagne plus grande n’est justifiée que si la loss robuste baisse et produit
au moins un gain de SSR sans franchir les gardes esthétiques.

La sonde reproductible construit les cinq sorties et réutilise strictement la
même source Stage 2 pour les deux raffinements :

```bash
python scripts/e020-srmpgd-robust-probe.py \
  --api http://127.0.0.1:18080 \
  --payload 'https://ptag.io/t/e020' \
  --launch
```

Le détail d’une vignette SR-MPGD affiche ensuite la table complète de sa
trajectoire. Le CSV contient aussi les colonnes `attempted_best_*`, afin de ne
pas confondre « aucune mise à jour retenue » et « aucune mise à jour calculée ».
