# E019 — SR-MPGD borné et sélection esthétique sûre

Date : 4 août 2026.

## Problème observé

E018 a prouvé que SRPG et SR-MPGD utilisaient enfin le même latent Stage 2. Il
n'a toutefois pas supprimé les taches lumineuses et colorées. L'audit du cœur
SR-MPGD a montré deux causes :

1. la mise à jour `gamma × gradient` n'avait aucune limite ;
2. le classement privilégiait tous les gains de lecture avant LPIPS, même quand
   l'image était visiblement détériorée.

Le papier donne `gamma=1000` et `lambda LPIPS=0,01`, mais ne publie pas le nombre
d'itérations. Ces valeurs dépendent également de la normalisation exacte de la
loss et du latent des auteurs. Elles ne peuvent pas être considérées comme une
recette universelle dans notre pile logicielle.

## Profil sûr par défaut

Le profil `diffqrcoder_srmpgd` utilise maintenant :

- 4 itérations au maximum ;
- gamma 100 ;
- LPIPS 0,10 ;
- pas latent RMS maximal 0,02 ;
- déplacement latent total maximal 0,06 ;
- gain QR relatif minimal 1 % ;
- LPIPS maximal 0,15 ;
- changement moyen de l'image maximal 6 % ;
- hausse de saturation moyenne maximale 4 % ;
- hausse des pixels très saturés maximale 5 % ;
- hausse des canaux RGB écrêtés maximale 1 %.

L'état zéro, c'est-à-dire le Stage 2 SRPG avant SR-MPGD, est toujours conservé
et reste sélectionnable.

## Algorithme de protection

À chaque itération :

1. le gradient brut et le pas `gamma × gradient` sont mesurés ;
2. le pas est réduit pour respecter la borne RMS ;
3. le déplacement cumulé est projeté dans la borne totale ;
4. l'image est décodée et comparée au SRPG initial ;
5. saturation, écrêtage, changement moyen, LPIPS et gain QR sont mesurés ;
6. un état hors garde est enregistré pour le diagnostic mais devient
   inéligible et arrête l'optimisation ;
7. parmi les états éligibles, le classement est : lecture stricte, SSR,
   robustesse par décodeur/scénario, LPIPS, changement d'image, MER puis perte
   SRL.

Ainsi, une image tachée ne peut plus gagner uniquement parce qu'un décodeur
supplémentaire la lit.

## Matrice factorielle

La recherche couvre :

- itérations : `1, 2, 4, 8, 20` ;
- gamma : `10, 30, 100, 300, 1000` ;
- LPIPS : `0,01, 0,05, 0,10, 0,25`.

Cela représente 100 configurations. Elles sont réparties en cinq campagnes,
une par gamma. Chaque campagne contient QR témoin, Stage 1, SRPG source et les
20 combinaisons itérations/LPIPS. Tous les SR-MPGD d'un lot réutilisent le même
latent SRPG et son SHA-256.

Ne lancer qu'un lot à la fois :

```powershell
python scripts/e019-srmpgd-grid.py `
  --api http://127.0.0.1:18080 `
  --payload https://ptag.io/t/e019 `
  --seeds 51001 `
  --launch 100
```

Sans `--launch`, le script écrit les cinq manifestes JSON sans lancer de GPU :

```powershell
python scripts/e019-srmpgd-grid.py `
  --api http://127.0.0.1:18080 `
  --payload https://ptag.io/t/e019
```

Commencer par le lot gamma 100 sur quatre prompts et une seed. Les autres lots
ne sont lancés qu'après vérification des SHA, de l'absence de taches et du temps
du premier lot.

## Valeurs à contrôler dans le Web Lab et le CSV

- `Appariement Stage 2 = Exact — SHA identique` ;
- `Pas latent RMS max` inférieur ou égal à la borne demandée ;
- `Garde esthétique = Respectée` ;
- `Gain QR suffisant = Oui` pour toute itération retenue après zéro ;
- `Arrêt SR-MPGD` ;
- itération retenue, LPIPS, déplacement latent et changement de l'image.

E019 ne promet pas encore 99 % de scans. Il empêche d'acheter un gain de scan
au prix d'une dégradation visuelle manifeste et produit les données nécessaires
pour choisir une recette robuste.
