# Catalogue de suivi

## Historique durable par génération

PostgreSQL conserve en production les éléments suivants ; SQLite utilise le même schéma pendant
les tests locaux :

| Groupe | Champs principaux |
|---|---|
| Run | identifiant, dates, backend, statut, prompt, hash du payload, seed, version QR |
| Performance | temps génération, validation et total, nombre de tentatives |
| Lecture | taux global, correspondance exacte, décodeur, scénario, latence |
| Structure | taux estimé de modules dont la luminance est incorrecte |
| Image | luminosité, contraste, entropie, netteté, pixels écrêtés |
| Tentative | seed, durées, taux de lecture, erreur modules, décision |
| Essai physique | terminal, OS, scanner, impression, matériau, taille, lumière, distance, angle, latence, résultat |

Chaque détail est disponible via l'API ; le tableau des runs peut être exporté en CSV.

## Perturbations simulées

| Scénario | Risque représenté |
|---|---|
| `original` | lecture numérique sans altération |
| `jpeg_90`, `jpeg_70` | recompression web/mobile |
| `blur_3` | mise au point et mouvement |
| `brightness_low`, `brightness_high` | éclairage insuffisant ou surexposition |
| `contrast_low` | impression terne ou lumière diffuse |
| `downscale_75` | redimensionnement et rééchantillonnage |
| `noise_gaussian` | bruit du capteur |
| `rotation_3` | défaut d'alignement |
| `perspective_mild` | prise de vue oblique |
| `print_dot_gain`, `print_dot_loss` | engraissement ou perte des points à l'impression |

Ces tests numériques ne remplacent pas une campagne physique avec imprimantes, matériaux,
dimensions, iPhone, Pixel et lecteurs industriels.

## Métriques Prometheus

| Série | Type | Usage |
|---|---|---|
| `prooftag_qr_runs_total` | compteur | acceptés, rejetés et erreurs par backend |
| `prooftag_qr_runs_active` | jauge | charge courante |
| `prooftag_qr_attempts` | histogramme | coût en nouvelles générations |
| `prooftag_qr_duration_seconds` | histogramme | latences génération, validation et totale |
| `prooftag_qr_validations_total` | compteur | résultat exact, mauvais payload ou non-détecté |
| `prooftag_qr_validation_duration_seconds` | histogramme | performance de chaque décodeur |
| `prooftag_qr_scan_pass_rate` | histogramme | robustesse des images finales |
| `prooftag_qr_module_error_rate` | histogramme | fidélité structurelle au QR source |
| `prooftag_qr_image_quality_latest` | jauge | dernière mesure de qualité d'image |
| `prooftag_qr_physical_validations_total` | compteur | résultats des scans terrain |

Les métriques DCGM existantes complètent le dashboard avec VRAM, utilisation GPU,
température et puissance.

## Indicateurs à ajouter pendant la phase FreeQR

- perte SRL globale et par type de module ;
- nombre de modules réparés et coût de réparation ;
- étape d'arrêt anticipé ;
- CLIPScore prompt-image ;
- score esthétique calibré ;
- LPIPS entre pré-réparation et image finale ;
- FID/KID sur les campagnes de comparaison, pas par requête ;
- succès par version QR, densité, niveau ECC, taille imprimée et terminal physique.
