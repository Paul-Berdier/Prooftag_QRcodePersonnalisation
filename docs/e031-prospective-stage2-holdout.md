# E031 — holdout prospectif Stage 2

## Statut et objet

Protocole figé le 21 août 2026. Aucun résultat E031 n'est revendiqué dans ce document. Le rapport
sera produit par `notebooks/26_e031_prospective_stage2_holdout.ipynb` après exécution. Modifier la
banque de prompts, les seeds, une recette, une porte ou la logique de sélection impose un nouvel
identifiant d'expérience.

E031 teste, sur des prompts jamais utilisés pour entraîner le conseiller, trois sorties Stage 2
SRPG :

1. recette fixe avec la seed A ;
2. chaîne Stage 1 + Stage 2 recommandée par prompt, avec la même seed A ;
3. recette fixe avec une seconde seed B.

Stage 1 est uniquement le parent exact de Stage 2 et n'est jamais livrable. SR-MPGD est exclu. Les
trois branches sont toutes générées avant l'analyse afin de conserver les contrefactuels appariés.
QR-Verify reste une mesure logicielle sur fichier : E031 ne mesure pas une probabilité de lecture
par téléphone et ne peut pas justifier une revendication de 99 %.

## Contrat exécutable

Les valeurs qui font foi sont également codées dans `prooftag_qr/e031_prospective.py` et copiées
dans le plan signé :

- 40 prompts : 20 simples et 20 atypiques ;
- payload par défaut : `https://ptag.io/t/e031` ;
- correction `M`, QR version 3, masque 4, module 20 px, padding 78 px ;
- seed A : `1_310_001` ; seed B : `1_310_002` ;
- Stable Diffusion 1.5 CetusMix Whalefall2, DiffQRCoder public et QR Code Monster v2, avec les
  révisions contrôlées au préflight ;
- sortie évaluée : `srpg` uniquement ; `srmpgd` interdit ;
- 5 répétitions du même scorer QR-Verify à 37 presets ;
- garde de saturation : maximum du taux de pixels fortement saturés et du taux de canaux RGB
  écrêtés, recalculé sur le raster téléchargé, `<= 0,05`.

Le `plan_id` est lié au plan logique, au registre de prompts, au SHA sémantique du conseiller, au
commit Git, à l'image OCI, à son digest et aux révisions de modèles. Un dossier d'un ancien runtime
ne peut donc pas être repris silencieusement.

Le hash sémantique des sources des cellules du notebook (sans sorties ni compteurs d'exécution)
et le SHA-256 du présent protocole sont eux aussi liés au plan. Ils sont revérifiés avant le
manifeste : modifier une cellule ou le protocole pendant l'expérience arrête le run.

Le CLIP utilisé pour les embeddings du conseiller est chargé avant le gel du plan. Sa révision
Hugging Face réelle doit être un SHA Git complet et le fichier
`sa_0_4_vit_b_32_linear.pth` est lié par SHA-256 ; l'absence de l'une de ces preuves arrête E031.

L'API GPU doit annoncer le même commit que le notebook, une image portant le tag des douze
premiers caractères de ce commit et un digest OCI complet. Le contrat qualité statique est gelé
dans le plan : CLIP ViT-B/32 à la révision `092a3b7e31726acc3a0207eea00f6040ac8b03a7`,
poids LAION aesthetic au SHA-256
`c7b14cead230694acc7b9447974d3cad78003c72da032e402a303b6c2429e85f`, et HPS v2.1 depuis
la source `866735ecaae999fa714bd9edfa05aa2672669ee3`, checkpoint révision
`697403c78157020a1ae59d23f111aa58ced35b0a`, SHA-256
`c57a38fb4a2f7e7c15bf00da2ea377cdf165448b4dd1052a484c215a998c9837`. L'état chargé ou non
du processus API n'entre pas dans le `plan_id` ; en revanche, chaque sortie Stage 2 doit exporter
les sept preuves effectives correspondantes, sinon l'expérience s'arrête.

## Banque de prompts réellement exécutée

Prompt négatif commun :

```text
text, letters, watermark, logo, barcode, oversaturated colors, blown highlights, muddy details
```

| ID | Famille | Prompt positif |
|---|---|---|
| `e031h_simple_001` | simple | A single amber glass chess knight on pale travertine, restrained product photograph, soft north light, generous negative space, no text or typography. |
| `e031h_simple_003` | simple | A folded indigo umbrella beside a rain-speckled window, quiet editorial photograph, natural materials, no text or typography. |
| `e031h_simple_005` | simple | A copper watering can on a weathered garden bench, balanced daylight photograph, one clear subject, no text or typography. |
| `e031h_simple_007` | simple | A pair of ivory ice skates on a muted blue floor, museum-like still life, controlled contrast, no text or typography. |
| `e031h_simple_009` | simple | A charcoal fountain pen resting on handmade cream paper, precise macro photograph, subtle shadows, no text or typography. |
| `e031h_simple_011` | simple | A yellow camping lantern inside a calm canvas tent, believable evening light, uncluttered composition, no text or typography. |
| `e031h_simple_013` | simple | A carved wooden duck on a grey linen shelf, understated catalogue photograph, coherent proportions, no text or typography. |
| `e031h_simple_015` | simple | A violet laboratory flask containing one fern leaf, clean studio photograph, soft reflections, no text or typography. |
| `e031h_simple_017` | simple | A small bronze handbell on dark green velvet, classic still life, focused warm light, no text or typography. |
| `e031h_simple_019` | simple | A coral-red table radio in a pale plywood alcove, modern editorial photograph, simple geometry, no text or typography. |
| `e031h_simple_021` | simple | A white porcelain mortar and pestle on black slate, calm culinary photograph, crisp material detail, no text or typography. |
| `e031h_simple_023` | simple | A moss-green binocular case on a sand-colored stool, outdoor equipment photograph, diffused daylight, no text or typography. |
| `e031h_simple_025` | simple | A translucent blue marble in a shallow wooden dish, minimal close-up photograph, realistic caustics, no text or typography. |
| `e031h_simple_027` | simple | A handwoven straw sunhat hanging on a limewashed wall, quiet summer photograph, gentle shadows, no text or typography. |
| `e031h_simple_029` | simple | A plum-colored violin bow in an open maple case, refined still life, controlled warm illumination, no text or typography. |
| `e031h_simple_031` | simple | A compact stainless steel moka pot on a terracotta tile, natural kitchen photograph, coherent reflections, no text or typography. |
| `e031h_simple_033` | simple | A single sea-green typewriter key displayed under glass, archival museum photograph, ample empty space, no text or typography. |
| `e031h_simple_035` | simple | A paper kite with orange tails leaning against a pale concrete wall, airy editorial photograph, no text or typography. |
| `e031h_simple_037` | simple | A black ceramic moon jar on a low oak plinth, gallery photograph, balanced symmetry, no text or typography. |
| `e031h_simple_039` | simple | A raspberry-red climbing helmet on folded canvas, equipment catalogue photograph, clean edges, no text or typography. |
| `e031h_atypical_002` | atypique | A miniature desert suspended inside a transparent cello, cinematic impossible still life, coherent glass and sand, intricate detail, no text or typography. |
| `e031h_atypical_004` | atypique | A library grown from interlocking seashells beneath green water, believable impossible architecture, controlled light, no text or typography. |
| `e031h_atypical_006` | atypique | A midnight carousel made of folded auroras orbiting a stone seed, detailed gouache scene, coherent geometry, no text or typography. |
| `e031h_atypical_008` | atypique | A silent brass orchestra nesting inside a giant pomegranate, surreal editorial photograph, natural material detail, no text or typography. |
| `e031h_atypical_010` | atypique | A glacier shaped as an open mechanical pocketbook crossing a lavender plain, cinematic scene, believable depth, no text or typography. |
| `e031h_atypical_012` | atypique | A translucent fox carrying a greenhouse of red moss on its back, museum diorama, coherent anatomy and reflections, no text or typography. |
| `e031h_atypical_014` | atypique | An underwater observatory built from stacked teacups and basalt, architectural visualization, controlled contrast, no text or typography. |
| `e031h_atypical_016` | atypique | A flock of ceramic umbrellas migrating through a candlelit tunnel, surreal photograph, consistent perspective, no text or typography. |
| `e031h_atypical_018` | atypique | A tiny railway station turning slowly inside a polished pearl, macro fantasy photograph, intricate coherent detail, no text or typography. |
| `e031h_atypical_020` | atypique | A canyon of blue fabric sewing itself around a levitating compass, cinematic impossible workshop, no text or typography. |
| `e031h_atypical_022` | atypique | A winter garden folded into the shadow of a copper key, poetic museum installation, believable lighting, no text or typography. |
| `e031h_atypical_024` | atypique | A glass lighthouse illuminating an inverted coral mountain above the clouds, restrained fantasy painting, no text or typography. |
| `e031h_atypical_026` | atypique | A clockwork heron assembling a river from silver ribbons, detailed editorial illustration, coherent motion, no text or typography. |
| `e031h_atypical_028` | atypique | A volcanic reading room balanced inside a hollow snowflake, architectural fantasy, controlled palette, no text or typography. |
| `e031h_atypical_030` | atypique | A procession of luminous mushrooms carrying a miniature suspension bridge, nocturnal diorama, natural textures, no text or typography. |
| `e031h_atypical_032` | atypique | An origami submarine cultivating an orchard of tiny moons, cinematic underwater scene, coherent impossible geometry, no text or typography. |
| `e031h_atypical_034` | atypique | A marble beehive projecting constellations into an empty theatre, museum installation photograph, no text or typography. |
| `e031h_atypical_036` | atypique | A velvet tornado carefully sorting porcelain fruit in a quiet pantry, surreal editorial scene, controlled contrast, no text or typography. |
| `e031h_atypical_038` | atypique | A floating observatory carved from black ice and inhabited by butterflies, atmospheric architectural rendering, no text or typography. |
| `e031h_atypical_040` | atypique | A mechanical lotus unfolding an entire rainy street from its petals, detailed cinematic scene, believable depth, no text or typography. |

## Gel du conseiller et contrôle de fuite

Le notebook charge les exports historiques explicitement listés, entraîne une seule fois
`E026ParameterAdvisor` avant toute génération, et enregistre le hash canonique des lignes, le hash
du pool effectif, le rapport de validation groupée par prompt, les hashes sémantique et fichier du
conseiller et toutes ses prédictions avant génération.

Tous les textes historiques et E031 sont normalisés en Unicode NFKC, casse et espaces. Toute
égalité exacte arrête le run. Le registre est exporté et hashé. E031 ne prétend pas détecter
automatiquement une paraphrase ou un sujet sémantiquement voisin : ce contrôle reste une limite
documentée, pas une promesse fictive.

Le conseiller choisit une chaîne Stage 1 puis Stage 2 conditionnée par le prompt. Il peut choisir
une configuration équivalente à la fixe si le pool appris ne contient pas de meilleur choix
distinct ; les signatures et rasters identiques restent visibles et ne sont jamais comptés comme
diversité. Une prédiction ne rend jamais une image livrable.

## Plan apparié et ordre d'exécution

| Branche | Stage 1 | Stage 2 | Seed | Rôle |
|---|---|---|---:|---|
| `fixed_seed_a` | fixe | fixe | `1_310_001` | premier essai |
| `advisor_seed_a` | conseillée | conseillée | `1_310_001` | paramètres conditionnés au prompt |
| `fixed_seed_b` | fixe | fixe | `1_310_002` | relance par nouvelle seed |

Chaque branche possède son Stage 1 exact et son Stage 2 :

```text
40 prompts × 3 branches × 2 états = 240 essais API
```

Les campagnes sont exécutées dans l'ordre déterministe du plan, par blocs de recette ; il n'y a
pas de carré latin dans E031-v1. C'est une limite contre les biais temporels, explicitement
conservée dans le rapport. La comparaison recette fixe/conseiller utilise le même prompt, le même
payload et la même seed A ; la comparaison des seeds utilise la même recette fixe. Chaque Stage 2
doit prouver le hash de son propre Stage 1.

## Reprise et intégrité

Le runner reprend uniquement une campagne distante portant le même plan et la même spécification.
Il refuse `completed_with_errors`, vérifie les exports CSV et les retélécharge s'ils manquent ou
sont corrompus. Les 240 états et les 120 Stage 2 doivent être présents. Pour chaque Stage 2,
l'audit exige le run source, le marqueur de réutilisation, le même prompt, la même seed, la même
campagne, la bonne méthode parente et le hash RGB exact du Stage 1.

Après téléchargement, le hash RGB du Stage 2 est recomputé. La saturation est elle aussi
recalculée localement sur ce raster ; une colonne API manquante ne devient jamais implicitement
zéro. QR-Verify reprend par hash de contenu avec un journal JSONL `fsync` et refuse un score lié à
un autre raster, payload, moteur ou protocole.

## Portes QR et politiques logicielles

Chaque raster Stage 2 unique est évalué cinq fois sur les mêmes 37 presets. Le payload doit être
exact dans chacune des cinq répétitions. La tolérance publiée est l'intersection des presets qui
réussissent dans toutes les répétitions.

| Porte | Condition | Usage |
|---|---|---|
| standard | au moins 30/37, payload exact 5/5, saturation `<= 0,05` | comparaison à E030 |
| stricte | au moins 36/37, payload exact 5/5, saturation `<= 0,05` | candidature finale |

Les politiques rejouées hors ligne sont : A seule, B seule, A puis B, A puis C, A puis B puis C,
et un `best_of_three` diagnostique. La cascade de production préenregistrée est A → B → C. Le
premier Stage 2 qui passe gagne ; sinon le prompt est rejeté. `best_of_three` n'est jamais présenté
comme un coût de production.

HPS v2.1, CLIPScore puis CLIP-Aesthetic ne servent qu'après la porte QR. Ils restent des proxys et
ne remplacent ni la présence du sujet, ni la discrétion du QR, ni un jugement humain.

## Revue humaine aveugle et porte finale

Les 120 Stage 2 sont mélangés de façon reproductible et présentés sans branche, seed, prédiction
ou score. Quatre images sont dupliquées sous un autre identifiant. Le CSV demande : esthétique,
fidélité au prompt, discrétion du QR et préférence globale de 1 à 5 ; présence du sujet, grille
trop visible et artefact fatal en oui/non ; lecture téléphone facultative, sans imputation.

La porte humaine est :

```text
esthétique >= 3
ET fidélité au prompt >= 3
ET discrétion du QR >= 3
ET sujet présent
ET grille pas trop visible
ET aucun artefact fatal
```

Le notebook exporte pour chacun des 120 candidats `software_deliverable`, `human_approved` et
`final_deliverable = software_deliverable ET human_approved`. Il rejoue ensuite chaque politique
avec la porte finale : si A échoue logiciellement ou humainement, la cascade essaie B puis C. Les
quatre doublons produisent les écarts absolus des notes et l'accord exact des réponses oui/non ;
ils ne comptent qu'une fois dans les taux.

## Analyse produite sans surpromesse

L'unité principale est le prompt. Le notebook produit :

- taux de livraison et IC Wilson 95 % de chaque politique, aux portes standard et stricte ;
- les mêmes taux séparés entre les 20 prompts simples et les 20 atypiques ;
- nombre moyen d'essais Stage 2 et d'essais API de chaque cascade ;
- moyennes de tolérance, saturation, HPS, CLIPScore et CLIP-Aesthetic des livrables ;
- répartition des branches sélectionnées ;
- graphiques par branche et politique ;
- notes humaines moyennes par branche, accord des doublons et cascade finale après revue ;
- nombre de rasters uniques, instabilités QR-Verify et inventaire complet des erreurs/provenances.

E031-v1 ne préenregistre pas de McNemar, bootstrap, test de corrélation ou analyse par sous-strate.
Ces analyses exigeraient un protocole/version dédié au lieu d'être ajoutées après observation.

## Règles stop/go

E031 peut autoriser une validation E032 plus large, jamais directement la production, si la
cascade A-B-C atteint au moins 38/40 à la porte QR stricte et au moins 36/40 à la porte finale
humaine, sans défaut de provenance. Les résultats simple/atypique doivent être inspectés
séparément. Un échec de cette règle n'autorise pas à modifier le seuil sur le même holdout.

Même 40/40 ne démontre pas 99 % de scannabilité physique : E031 n'inclut ni téléphone ni
impression et son intervalle statistique reste trop large.

## Artefacts obligatoires

- protocole et registre de prompts ;
- plan public, prédictions, état du runner et copies des exports CSV ;
- résultats des 240 états et audit d'appariement des 120 Stage 2 ;
- galerie et hashes RGB/PNG ;
- journal QR-Verify, tableau par raster, décisions et synthèses des politiques ;
- audit des pins qualité effectifs pour chacun des 120 rasters Stage 2 ;
- revue aveugle, révélation, accord des doublons et décisions finales lorsqu'elle est remplie ;
- rapport Markdown/JSON, graphes, manifeste de versions et checksums ;
- archive `.tar.gz` et sidecars SHA-256 copiés dans `/workspace/downloads`.

Le rapport de résultats sera ajouté dans un document daté après exécution. Ce protocole ne doit
pas être réécrit pour faire correspondre les portes aux observations.
