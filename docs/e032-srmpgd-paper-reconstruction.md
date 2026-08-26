# E032 — reconstruction contrôlée du SR-MPGD du papier

## Pourquoi E032 est nécessaire

Les conclusions E019, E027 et E029 concernent le SR-MPGD **sécurisé** de Prooftag. Elles ne
réfutent pas les équations 12 à 14 de DiffQRCoder. Cette variante locale utilisait notamment quatre
itérations, `gamma=100`, `LPIPS=0,10`, une porte MER, un plafond RMS du pas, une limite de déplacement
total, des gardes esthétiques, un arrêt QR-Verify et un oracle de sélection. Plusieurs de ces
mécanismes peuvent conserver l'état zéro ou rendre différents gamma numériquement équivalents.

Le papier rapporte au contraire des gains de 5 à 12 points de SSR après SR-MPGD. Par exemple, pour
`lambda1=500` et `lambda2=3`, la Table 7 passe de 89 % à 99 %. Les hyperparamètres publiés du
post-traitement sont `gamma=1000` et `lambda_LPIPS=0,01`. Le nombre d'itérations n'est pas publié.

## Écarts causaux découverts

1. Le profil Prooftag actif part d'un Stage 2 partiel à cible binaire, et non du Stage 2 complet
   QArt utilisé pour la Table 7.
2. En mode QArt, le backend transmettait le blueprint QArt à SR-MPGD. Les équations imposent la
   cible QR originale `y`; `y_tilde` ne sert qu'au ControlNet/SRPG du Stage 2.
3. La porte MER pouvait interdire toute descente avant la première mise à jour.
4. Le plafond `max_step_rms=0,02` compensait automatiquement `gamma`. Une fois le plafond atteint,
   `gamma=10`, `100` ou `1000` pouvait produire le même pas appliqué.
5. La sélection QR-Verify et les gardes esthétiques pouvaient restaurer l'état zéro, alors que le
   papier ne publie aucun oracle de sélection.
6. Le dépôt public n'est pas une implémentation complète du papier : QArt est absent, le Stage 2
   peut repartir d'un bruit neuf, la référence perceptuelle SR-MPGD reste le Stage 1, l'objectif
   réemploie les poids SRPG et le paramètre `srmpgd_lr` n'est pas correctement transmis.

## Deux protocoles qui ne doivent plus être confondus

### `guarded_production`

Conserve le comportement E019 : précondition MER, caps latents, gardes esthétiques, arrêt strict et
sélection du meilleur candidat externe. Ce mode limite les taches, mais ne mesure pas fidèlement
l'ablation scientifique.

### `paper_equations`

Exécute directement :

```text
z_i = z_(i-1) - gamma * grad_z(
    SRL(VAE.decode(z_(i-1)), QR_original)
    + 0.01 * LPIPS(VAE.decode(z_(i-1)), image_Stage2_initiale)
)
```

Ce mode :

- reprend le latent propre exact du Stage 2 ;
- cible toujours le QR original, même si le Stage 2 utilise QArt ;
- utilise la SRL locale reconstruite depuis les équations 1 à 6 ;
- n'applique ni porte MER, ni cap RMS, ni cap de déplacement total ;
- n'applique ni arrêt esthétique, ni arrêt QR-Verify ;
- conserve toutes les itérations et retourne l'itération finale fixée ;
- reste désactivé par défaut et ne constitue jamais une politique de livraison.

## Profil Web Lab

Les profils désactivés `diffqrcoder_paper_srmpgd` et
`diffqrcoder_paper_srmpgd_guarded` partagent exactement le même Stage 2. Le premier utilise :

- Stage 1 et Stage 2 de 40 pas ;
- initialisation Stage 2 depuis le latent du Stage 1 rebruité ;
- force Stage 2 `1,0` ;
- SRG `500`, PG `3` ;
- approximation QArt publique à fragment d'URL pour le Stage 2 ;
- QR original pour SR-MPGD ;
- 20 itérations exploratoires, `gamma=1000`, LPIPS `0,01`.

Le second conserve les mêmes 20 itérations, gamma et poids LPIPS, mais réactive les plafonds,
gardes, arrêts et la sélection externe. Cette paire isole leur effet sans modifier le latent parent.

Le QArt exact des auteurs n'est pas public. Le profil est donc une reconstruction des équations avec
un proxy QArt public, pas une revendication de reproduction intégrale.

## Incident du premier plan E032

Le premier plan, `48ed7ac799e61502`, s'est terminé avec **30 campagnes sur 30 en
`completed_with_errors`**. Ce résultat ne mesure pas le SR-MPGD : il provient d'une erreur de
géométrie commune à tous les contextes.

Le mode automatique de découpe retirait 80 pixels de chaque côté du raster VAE de 736 pixels. Le
cœur obtenu mesurait donc 576 pixels, alors que le QR version 3 contient 29 modules par côté :

```text
736 - 2 * 80 = 576
576 / 29 = 19,862... pixels par module
```

Cette géométrie non entière est incompatible avec la SRL, qui doit associer chaque cellule du QR à
un nombre entier de pixels. L'erreur racine est donc le refus
`QR core does not have an integer module geometry`, avant toute itération interprétable de
SR-MPGD. Un contrôle de hash exécuté dans le chemin d'échec a ensuite produit l'erreur la plus
visible dans certains résumés et a masqué ce premier défaut. Le contrôle d'intégrité n'était pas la
cause de l'échec scientifique ; il était une erreur secondaire de remontée/diagnostic.

La géométrie E032 corrigée est désormais explicite et ne dépend plus du crop automatique :

```text
padding = 78 pixels
736 - 2 * 78 = 580
580 = 29 modules * 20 pixels
```

Les profils `paper_equations` et `guarded_production` doivent tous deux employer
`srmpgd_crop_padding_px=78`. Ce paramètre appartient au contrat expérimental : le modifier exige un
nouveau plan.

### Diagnostic et conservation des preuves

Le notebook collecte les CSV de toutes les campagnes, y compris celles marquées
`completed_with_errors`, avant de décider si l'analyse peut continuer. Le diagnostic :

- publie les statuts par méthode et les messages d'erreur complets, sans troncature ;
- regroupe les signatures d'erreur afin de distinguer la cause géométrique du contrôle de hash
  secondaire ;
- télécharge toutes les images réellement produites par les essais `accepted` ou `rejected` qui
  possèdent un `generation_run_id` ;
- génère des planches-contact par prompt et seed ;
- conserve les exports, le JSON de diagnostic et les images dans l'archive
  `48ed7ac799e61502-e032-diagnostic.tar.gz` lorsqu'elle est disponible.

Ces artefacts restent utiles pour l'audit de l'incident et ne doivent pas être supprimés. En
revanche, ils ne constituent ni une matrice appariée complète, ni des observations valides du
mécanisme SR-MPGD.

### Décision de reprise

Le plan `48ed7ac799e61502` est **invalide scientifiquement** et ne doit jamais être repris, complété,
fusionné avec des résultats corrigés ou utilisé pour calculer un taux de succès. La correction du
crop modifie le contrat d'exécution ; une nouvelle campagne doit donc obtenir un nouveau
`plan_id`, un nouveau dossier de reprise et de nouveaux exports. L'ancien dossier est conservé en
lecture seule comme preuve d'incident.

## Campagne appariée à exécuter

Avant toute campagne longue :

1. geler dix prompts (cinq simples, cinq atypiques), trois seeds et le même payload court ;
2. générer un seul Stage 1 et un seul latent Stage 2 par contexte ;
3. comparer, sur ce même latent, l'état 0, `paper_equations` et `guarded_production` ;
4. sauvegarder au minimum les états 0, 1, 2, 4, 8, 12 et 20 ;
5. rapporter séparément l'itération finale et le meilleur état rétrospectif ;
6. mesurer SRL, MER, QR-Verify original, validation robuste, RMS gradient/pas, delta latent,
   LPIPS, saturation, écrêtage, CLIP-Aesthetic, CLIPScore et HPS ;
7. ne livrer aucune sortie de ce profil automatiquement.

Le test est concluant seulement si la SRL décroît réellement, si `gamma` modifie le pas appliqué et
si QR-Verify s'améliore sur plusieurs contextes sans effondrement systématique des métriques
visuelles. Un échec de ce protocole restera à interpréter avec la limite QArt publique.
