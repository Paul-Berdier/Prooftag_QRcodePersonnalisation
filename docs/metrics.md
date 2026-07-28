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
| `prooftag_qr_guided_rediffusions_total` | compteur | résultat de la seconde diffusion : porte qualité franchie, rejet de préservation ou erreur |
| `prooftag_qr_guided_rediffusion_duration_seconds` | histogramme | coût de la seconde passe img2img/ControlNet |
| `prooftag_qr_guided_rediffusion_module_error_rate` | jauge | erreur module avant et après la seconde diffusion |
| `prooftag_qr_guided_rediffusion_image_change` | jauge | pixels modifiés et changement absolu moyen de la seconde diffusion localisée |
| `prooftag_qr_srpg_runs_total` | compteur | sortie SRPG acceptée par la porte réelle, rejetée avec motif ou erreur |
| `prooftag_qr_srpg_duration_seconds` | histogramme | coût de la boucle DDIM différentiable complète |
| `prooftag_qr_srpg_module_error_rate` | jauge | erreur module réelle de l'image avant/après SRPG |
| `prooftag_qr_srpg_step_diagnostic` | jauge par pas/métrique | erreur centrale, SRL, LPIPS, gradient RMS et delta de bruit RMS des 40 pas |
| `prooftag_qr_srpg_image_change` | jauge | pixels modifiés et MAE de la sortie SRPG |
| `prooftag_qr_srpg_peak_gpu_memory_allocated_mib` | jauge | pic CUDA alloué par PyTorch pendant SRPG |
| `prooftag_qr_srpg_gradient_clips_total` | compteur | pas dont le delta de bruit atteint sa borne de sécurité |
| `prooftag_qr_latent_refinements_total` | compteur | convergence, amélioration acceptée, rejet par préservation, absence d'amélioration ou erreur du raffinement latent |
| `prooftag_qr_latent_refinement_duration_seconds` | histogramme | coût du raffinement VAE/SRL |
| `prooftag_qr_latent_refinement_iterations` | histogramme | nombre d'itérations réellement exécutées |
| `prooftag_qr_latent_refinement_module_error_rate` | jauge | erreur des sous-modules centraux avant/après et meilleur résultat observé |
| `prooftag_qr_latent_refinement_loss` | jauge | composantes SRL, préservation et changement absolu moyen, retenues et observées |
| `prooftag_qr_lab_campaigns_total` | compteur | campagnes Web par état terminal |
| `prooftag_qr_lab_campaigns_active` | jauge | campagne Web actuellement exécutée |
| `prooftag_qr_lab_trials_total` | compteur | essais Web par méthode et résultat |
| `prooftag_qr_lab_trial_duration_seconds` | histogramme | durée de bout en bout par méthode |
| `prooftag_qr_lab_ratings_total` | compteur | évaluations humaines enregistrées |
| `prooftag_qr_lab_quality_scores_total` | compteur | calculs CLIP/esthétique réussis ou en erreur |
| `prooftag_qr_lab_quality_score_duration_seconds` | histogramme | durée CPU de CLIPScore et CLIP-aesthetic |

Les journaux `repair_variant_validated` contiennent aussi la tentative, la seed, toutes les
métriques visuelles et la liste exacte des scénarios en échec. Le benchmark les exporte dans
`variants.csv` et `variant-failures.csv`.

Les variantes préfixées `latent_` sont des réparations ciblées calculées depuis la sortie SRL,
par exemple `latent_rounded_16`. Les variantes sans ce préfixe restent calculées depuis l'image
brute et constituent la chaîne de secours. Le nom de la variante sélectionnée doit être contrôlé
avant d'attribuer un changement de `final.png` au raffinement latent.

Avec E004, `guided_*` désigne une sortie issue de la seconde diffusion et
`guided_latent_*` une sortie ayant ensuite reçu SR-MPGD. Les artefacts `guided_control` et
`guided_mask`, `guided_unprojected` et `guided_projected` sont diagnostiques et ne sont jamais
sélectionnables comme résultat final. Ils permettent de distinguer l'effet global de la seconde
diffusion de sa projection locale avant validation.

Avec E005, `srpg` est toujours envoyé aux 26 validations, indépendamment de sa porte interne.
La porte décide seulement si les réparations `srpg_*` peuvent repartir de cette image. Le journal
`srpg_completed` conserve les 40 diagnostics, la décision, le motif de rejet et la VRAM ; le
benchmark les normalise dans `srpg-steps.csv`.

Les métriques DCGM existantes complètent le dashboard avec VRAM, utilisation GPU,
température et puissance. Le laboratoire expose en plus les agrégats de campagne dans son
interface et un export CSV persistant ; Prometheus reste destiné à la supervision temporelle.

## Indicateurs ajoutés par la campagne E007

- CLIPScore prompt-image et similarité CLIP brute ;
- CLIP-aesthetic LAION ;
- résultat par axe prompt, seed et payload ;
- risque du brut : erreur fonctionnelle/data, marge module, entropie et densité de contours ;
- paramètres TPE complets et importance par rapport à l'objectif de scan ;
- prédiction et incertitude du mini-modèle ;
- MAE de validation croisée groupée par contexte.

Ces mesures sont persistées dans les artefacts E007 avant leur éventuelle exposition Prometheus.

## Indicateurs ajoutés par la campagne E008

- identifiant exact du ControlNet, sous-dossier de poids et profil de conditionnement ;
- taux de lecture brut et après SRPG, séparés pour chaque contexte ;
- pire taux de lecture, moyenne et porte 26/26 sur la campagne complète ;
- CLIP-aesthetic, CLIPScore, erreur module, durée et pic VRAM ;
- erreurs de chargement persistées, groupe incomplet explicitement non promouvable ;
- graphique comparatif du compromis scan/esthétique avant et après SRPG.

Le vainqueur automatique E008 n'autorise pas une mise en production : il fixe seulement le
ControlNet à réoptimiser dans E007. La livraison reste soumise aux holdouts puis aux scans
physiques multi-téléphones et imprimés.

## Indicateurs restant à ajouter pendant la phase d'entraînement

- perte SRL par type de module sur la durée d'une campagne ;
- proportion exacte de modules modifiés par la réparation (la proportion de pixels est déjà suivie) ;
- étape d'arrêt anticipé ;
- LPIPS entre pré-réparation et image finale ;
- FID/KID sur les campagnes de comparaison, pas par requête ;
- succès par version QR, densité, niveau ECC, taille imprimée et terminal physique.
