# E024 — QR-Verify comme autorité logicielle unique

## Décision

À partir d'E024, la validation automatique des images du Web Lab repose
uniquement sur [`antfu/qr-verify`](https://github.com/antfu/qr-verify), version
`0.2.0`. Son scanner est `qr-scanner-wechat` `0.1.3`, c'est-à-dire
l'implémentation WeChat QR exécutée en WebAssembly.

OpenCV, ZBar, ZXing, l'ancien SSR et le phone proxy ne participent plus au
verdict automatique. Depuis E025, CLIP-AES, CLIPScore et HPS sont de nouveau
calculés, mais restent strictement séparés de l'acceptation QR. L'appréciation
esthétique humaine reste enregistrée dans le Web Lab.

## Adaptation Prooftag

Le CLI amont répond à la question « un texte quelconque a-t-il été décodé ? »
et mélange aléatoirement ses prétraitements. Ce n'est pas suffisant pour un QR
Prooftag. Le pont local conserve le scanner, la réduction à 300 px et la matrice
de tolérance amont, mais :

1. exécute les 37 presets dans un ordre déterministe ;
2. recrée une image Sharp indépendante pour chaque preset ;
3. renvoie tous les textes décodés à Python ;
4. exige le payload attendu byte-à-byte, ou l'URL canonique sans fragment pour
   l'expérience QArt explicitement marquée ;
5. échoue ouvertement si Node, le WASM ou l'un des 37 résultats manque.

Les 37 presets sont l'image grisée sans filtre, puis les 36 combinaisons de
contraste `6/3/1,5`, luminosité `0,9/1,2/1,4` et flou `0,5/1/1,5/2` du projet
amont.

## Mesures et décision

- **QR-Verify valide** : au moins un des 37 presets restitue le payload attendu.
- **Lecture sans filtre** : le preset original restitue le payload attendu.
- **Score QR-Verify** : nombre de presets exacts divisé par le nombre de presets
  que le QR témoin sait lui-même passer.
- **MER** : diagnostic géométrique conservé, sans rôle de décodeur.

Le score est utile pour classer la tolérance logicielle, mais ce n'est pas une
probabilité de lecture par téléphone. Les scans manuels restent la vérité terrain
tant qu'un banc caméra n'est pas disponible.

## Versions et sécurité

- `qr-verify@0.2.0` ;
- `qr-scanner-wechat@0.1.3` ;
- Node.js `22.15.0` dans l'image Docker ;
- `sharp@0.35.3`, forcé à la place de l'ancienne version vulnérable transitivement
  demandée par QR-Verify ;
- lockfile npm commité ;
- `npm audit --omit=dev` : aucune vulnérabilité connue au 5 août 2026.

Le build Docker exécute un vrai décodage WASM d'un QR témoin et exige `37/37`.

## Re-test contrôlé

Le script `scripts/e024-qr-verify-retest.py` reprend sans les modifier les dix
prompts E022, la seed `61001` et les deux recettes DiffQRCoder appariées. Seul
le système de mesure change. Il vérifie le contrat E024 exposé par l'API avant
de créer la campagne.

```bash
python scripts/e024-qr-verify-retest.py \
  --api http://127.0.0.1:18080 \
  --payload https://ptag.io/t/e024 \
  --launch
```

Le Web Lab sert ensuite à noter l'esthétique et les scans téléphone image par
image. Comparer E023 et E024 permet de mesurer le changement de validateur, pas
une amélioration du générateur.
