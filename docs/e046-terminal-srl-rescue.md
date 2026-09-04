# E046 — reprise après divergence SRL officielle

## Erreur traitée

```text
local upstream SRL port diverged from the pinned official class
```

Cette erreur ne signifie pas qu'un QR a échoué à WeChat. Elle signifie que la
fonction de perte SRL locale et la classe officielle épinglée ne donnent plus la
même valeur avec la tolérance scientifique exigée.

La branche concernée ne doit donc pas être forcée en relâchant silencieusement la
tolérance. Elle est :

- classée `scientific_fidelity_mismatch` ;
- marquée non réessayable ;
- exclue du scoring final ;
- conservée dans les traces d'échec ;
- remplacée par les autres recettes et le parent Stage2 déjà disponibles.

## Reprise

Le script utilise l'image Docker et le commit qui ont créé le plan existant. Il
ne reconstruit aucun parent et ne recommence aucune trajectoire réussie.

```bash
bash scripts/resume-e046-terminal-srl.sh
```

Le script :

1. détecte les échecs SRL déjà persistés ;
2. crée des marqueurs sous `terminal-refinements/` ;
3. poursuit tous les refinements encore manquants ;
4. met automatiquement en quarantaine toute nouvelle divergence identique ;
5. score uniquement les trajectoires terminées ;
6. agrège les résultats, produit les QR finaux par prompt et vérifie le manifeste.

## Ce qui n'est pas modifié

- aucune image existante ;
- aucun latent existant ;
- aucun score WeChat existant ;
- aucune tolérance SRL ;
- aucun résultat réussi ;
- aucun fichier `/data` supprimé.
