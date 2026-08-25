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
