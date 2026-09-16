# Points d'intégration exacts et limites

Base scientifique lue via GitHub : bfa4cdce4c6333e5214af6f3468a24a5374302c1.
Aucun fichier de ce cœur n'est remplacé par le ZIP.

## Scoring d'une campagne partielle

`worker.inventory` appelle `e046_large_campaign.load_plan` puis `_promotion_valid`
avec `PARENT_GENERATION_REQUIRED` et vérification profonde. Il n'appelle ni
`run-e046-large-dataset.sh resume`, ni `generate_parent`, ni l'agrégation historique.

Pour un parent présent : `_settings_for_plan`, `_quality_scores`, `_score_qr_cached`,
`_stage2_parent_row`. Le scorer est le `ConservativeQRVerifyScorer` déjà utilisé,
avec contrat vérifié par `_new_qr_scorer`. Les caches de ce nouveau scoring sont
placés dans le nouveau run. Les scores historiques ne sont réutilisés que s'ils sont
promus, du même contrat et de la même empreinte d'image. Rien n'est écrit dans /data.

## Entraînement

Le nouveau module `learning.py` consomme les observations parentes dérivées. Il ne
force pas un dataset E046 smoke à se déclarer prêt et ne modifie pas les autorisations
historiques. L'autorisation humaine est celle du lancement explicite de ce run E047.

Les partitions sont calculées avant recherche des hyperparamètres et persistées.
L'encodeur et la sélection sont réajustés dans chaque pli. Le modèle retenu est
réajusté sur la partie fit, calibré sur calibration, puis testé sur test. Aucun
réajustement sur le test n'est réalisé. Les régressions ne partagent pas un
transformer réajusté qui casserait le classifieur. Il n'y a pas de promesse de gain.

Limites : le plan de collecte est partiel et peut avoir une couverture biaisée ;
TF-IDF n'est pas un embedding sémantique ; les proxys visuels ne remplacent pas les
notes humaines ; le payload humoristique est une variation de distribution. Aucun
résultat de leave-one-family-out n'est inventé : cette version mesure un test par
groupes et une validation croisée groupée, pas une étude complète de généralisation.

## Génération

`UpstreamDiffQRCoderBackend.generate` et `_run_stage2` servent à générer les deux
étapes. Les aperçus sont capturés par le callback natif `stage2_x0_estimate_step_*`.
Le latent exact est sauvegardé avec safetensors. Les variantes de peinture de
quiet zone et les réparations de bordure ne sont pas appelées.

Référence fixée : `PARENT_RECIPES[4]` = m4_e044_anchor.
Référence proche papier : `PARENT_RECIPES[7]` = m7_paper_strong.
Leur sémantique complète est enregistrée dans chaque task.json. La deuxième n'est
pas rebaptisée reproduction exacte : cible binaire exacte et wrapper public restent
des différences documentées par rapport au papier.

## Opérations

Les fichiers d'opération sont root-owned sous /var/lib/prooftag-qr-nightly.
Le code du superviseur est copié sous /opt dans un dossier adressé par son contenu.
Les workers de calcul ne reçoivent pas de kubeconfig ni de ServiceAccount token.
Le gardien ne dépend ni de scikit-learn ni de PyTorch ni du conteneur de calcul.

Une restauration réussie signifie : réplicas vLLM rétablis, rollout disponible,
/health et /v1/models accessibles, petite inférence réussie, même modèle annoncé.
Ce n'est pas un test complet de la connexion Entra, du RAG, du Web ou des autres outils.
Un autre opérateur modifiant le déploiement ou une panne Kubernetes peut nécessiter
une intervention humaine. La limite 08:00 est un objectif protégé par budgets et
restauration anticipée, pas une garantie matérielle.
