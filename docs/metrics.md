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
| `prooftag_qr_regenerations_total` | compteur | nouvelles diffusions déclenchées avant le fallback global |
| `prooftag_qr_duration_seconds` | histogramme | latences génération, validation et totale |
| `prooftag_qr_validations_total` | compteur | résultat exact, mauvais payload ou non-détecté |
| `prooftag_qr_validation_duration_seconds` | histogramme | performance de chaque décodeur |
| `prooftag_qr_scan_pass_rate` | histogramme | robustesse des images finales |
| `prooftag_qr_module_error_rate` | histogramme | fidélité structurelle au QR source |
| `prooftag_qr_image_quality_latest` | jauge | dernière mesure de qualité d'image |
| `prooftag_qr_physical_validations_total` | compteur | résultats des scans terrain |
| `prooftag_qr_model_loads_total` | compteur | chargements ControlNet réussis ou en erreur |
| `prooftag_qr_model_load_duration_seconds` | histogramme | durée du chargement initial par résultat |
| `prooftag_qr_model_loaded` | jauge | présence du pipeline ControlNet en VRAM |
| `prooftag_qr_repair_variants_total` | compteur | acceptation de chaque profil de réparation |
| `prooftag_qr_repair_selected_total` | compteur | profil finalement retenu pour l'image |
| `prooftag_qr_repair_variant_scan_pass_rate` | jauge | dernier taux de lecture de chaque variante |
| `prooftag_qr_repair_variant_module_error_rate` | jauge | dernière erreur modules de chaque variante |
| `prooftag_qr_repair_variant_image_quality` | jauge | qualité visuelle et écart à l'image brute de chaque variante (`changed_pixel_ratio`, `mean_absolute_change`, entropie, écrêtage, contraste, luminosité et netteté) |
| `prooftag_qr_latent_refinements_total` | compteur | convergence, amélioration, absence d'amélioration ou erreur du raffinement latent |
| `prooftag_qr_latent_refinement_duration_seconds` | histogramme | coût du raffinement VAE/SRL |
| `prooftag_qr_latent_refinement_iterations` | histogramme | nombre d'itérations réellement exécutées |
| `prooftag_qr_latent_refinement_module_error_rate` | jauge | erreur des sous-modules centraux avant/après |
| `prooftag_qr_latent_refinement_loss` | jauge | dernières composantes SRL et préservation |

Les journaux `repair_variant_validated` contiennent aussi la tentative, la seed, toutes les
métriques visuelles et la liste exacte des scénarios en échec. Le benchmark les exporte dans
`variants.csv` et `variant-failures.csv`.

Les métriques DCGM existantes complètent le dashboard avec VRAM, utilisation GPU,
température et puissance.

## Indicateurs restant à ajouter pendant la phase d'entraînement

- perte SRL par type de module sur la durée d'une campagne ;
- proportion exacte de modules modifiés par la réparation (la proportion de pixels est déjà suivie) ;
- étape d'arrêt anticipé ;
- CLIPScore prompt-image ;
- score esthétique calibré ;
- LPIPS entre pré-réparation et image finale ;
- FID/KID sur les campagnes de comparaison, pas par requête ;
- succès par version QR, densité, niveau ECC, taille imprimée et terminal physique.
