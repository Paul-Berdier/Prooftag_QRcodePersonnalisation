# DiffQRCoder - traduction expliquée en français

## Comment fabriquer un QR code artistique qui reste facile à scanner ?

Traduction pédagogique de l'article **« DiffQRCoder: Diffusion-based Aesthetic QR Code Generation with Scanning Robustness Guided Iterative Refinement »**, version 3 du 15 février 2025, par Jia-Wei Liao et ses collègues.

> Cette version est une traduction fidèle mais reformulée pour être comprise au niveau lycée. Elle conserve les idées, les résultats, les limites et le sens des équations. La longue liste bibliographique n'est pas retraduite, car elle contient surtout des titres et des noms propres.

---

## L'essentiel en deux minutes

Un QR code ordinaire est très fiable, mais pas très joli. Une intelligence artificielle peut le transformer en paysage, en bâtiment ou en illustration. Le problème est qu'en embellissant les petits carrés, elle risque de rendre le code impossible à scanner.

Les auteurs proposent **DiffQRCoder**, une méthode qui cherche deux objectifs en même temps :

1. produire une belle image correspondant à une consigne écrite ;
2. conserver assez bien la structure du QR code pour qu'un téléphone puisse le lire.

Leur système travaille en deux grandes étapes :

- il crée d'abord une image artistique à partir du QR code et du texte demandé ;
- il corrige ensuite progressivement les zones importantes pour la lecture, tout en essayant de ne pas dégrader l'image.

Le point important est que le système ne demande pas d'entraîner un nouveau modèle. Il utilise des modèles déjà entraînés, notamment Stable Diffusion, ControlNet et un VAE, puis les guide avec une nouvelle fonction d'erreur conçue spécialement pour les QR codes.

Dans leurs expériences, les auteurs annoncent un **taux de lecture de 99 %**, contre 60 % pour leur génération ControlNet sans correction. Leurs QR codes restent lisibles à 97 % avec une inclinaison simulée de 45 degrés et à 96 % avec le niveau de correction d'erreur le plus faible testé. La méthode n'est cependant pas parfaite : elle peut encore échouer, elle demande de régler plusieurs paramètres et elle utilise parfois une étape supplémentaire de post-traitement.

---

## Petit dictionnaire avant de commencer

**Module** : un des petits carrés noirs ou blancs qui composent un QR code. Un module peut contenir de nombreux pixels dans l'image affichée.

**Pixel** : le plus petit point coloré d'une image numérique.

**Modèle de diffusion** : une IA qui apprend à créer une image en partant d'un bruit aléatoire, puis en retirant progressivement ce bruit.

**Prompt** : la consigne textuelle donnée à l'IA, par exemple « cascade majestueuse dans une forêt tropicale ».

**ControlNet** : un outil qui permet de guider Stable Diffusion avec une structure visuelle, ici celle du QR code.

**Espace latent** : une représentation mathématique compressée de l'image. L'IA travaille souvent dans cet espace, plus petit et plus pratique que l'image complète.

**VAE** : un réseau qui transforme une image en représentation latente, puis peut reconstruire une image à partir de cette représentation.

**Fonction de perte** : un nombre qui mesure à quel point le résultat est éloigné de l'objectif. Plus ce nombre est grand, plus l'algorithme doit corriger l'image.

**Gradient** : une indication mathématique donnant la direction dans laquelle modifier le résultat pour diminuer une erreur.

**SSR, Scanning Success Rate** : pourcentage de tentatives où le QR code est lu avec succès.

**LPIPS** : mesure de ressemblance visuelle inspirée de la perception humaine. Elle compare des caractéristiques de haut niveau, et pas seulement les pixels un par un.

---

# Traduction expliquée de l'article

## Résumé de l'article

Les modèles de diffusion ont beaucoup amélioré la génération d'images et ont aussi transformé la création de QR codes artistiques. Mais lorsqu'un QR code devient plus beau, sa capacité à être scanné est souvent sacrifiée, ce qui limite son utilisation réelle.

Pour répondre à ce problème, les auteurs proposent **DiffQRCoder**, un générateur de QR codes fondé sur la diffusion et ne nécessitant pas de nouvel entraînement. Il doit fabriquer des codes à la fois lisibles et agréables à regarder.

La première nouveauté est le **Scanning-Robust Perceptual Guidance**, ou **SRPG**. C'est un mécanisme qui guide le modèle pendant qu'il enlève le bruit. Il pousse l'image à respecter le véritable QR code, tout en conservant son apparence artistique.

La seconde nouveauté est un post-traitement nommé **Scanning-Robust Manifold Projected Gradient Descent**, ou **SR-MPGD**. Il corrige encore l'image dans son espace latent afin d'augmenter sa robustesse à la lecture.

D'après les expériences, DiffQRCoder obtient un meilleur taux de lecture que les méthodes comparées, avec une qualité esthétique meilleure ou comparable. Par rapport à ControlNet utilisé seul, le taux de lecture passe de 60 % à 99 %. Une étude auprès de participants indique également que les images sont jugées attirantes. Même avec des angles de lecture différents ou des réglages de correction d'erreur stricts, le taux dépasse 95 % dans les tests correspondants.

### En langage simple

L'IA reçoit deux choses : le QR code d'origine et une description de l'image désirée. Elle dessine d'abord librement, puis un « professeur de QR code » lui signale les carrés importants qui risquent d'être mal lus. Elle retouche ces carrés sans effacer complètement le dessin.

---

## 1. Introduction

Les QR codes sont utilisés partout : paiements, partage d'informations et publicité. Ils sont rapides à lire et presque tous les smartphones possèdent un scanner. Leur défaut est surtout visuel : une grille noire et blanche s'intègre mal dans une affiche ou un produit.

Les QR codes artistiques peuvent attirer l'attention, mieux s'intégrer à un design et renforcer une identité de marque. Plusieurs méthodes ont donc été créées pour les embellir.

Certaines anciennes techniques utilisent le **transfert de style** : elles mélangent les textures d'une image avec la structure du QR code. Elles manquent parfois de souplesse et peuvent réduire la fiabilité du scan.

Les outils plus récents emploient des modèles génératifs comme Stable Diffusion et ControlNet. On peut régler la force avec laquelle l'IA suit le texte ou la forme du QR code, mais il existe un conflit :

- si on impose trop fortement le QR code, le résultat reste lisible mais paraît moins naturel ;
- si on laisse plus de liberté artistique, l'image est plus belle mais le code risque de ne plus fonctionner.

Dans les usages commerciaux, des personnes corrigent parfois manuellement les codes illisibles. Cette opération est lente. Le défi consiste donc à obtenir automatiquement un bon équilibre entre beauté et robustesse.

Les auteurs présentent trois contributions principales :

1. un système en deux étapes, sans nouvel entraînement, avec le guidage SRPG ;
2. un post-traitement SR-MPGD qui peut porter le taux de lecture jusqu'à 100 % dans certaines configurations expérimentales ;
3. des tests quantitatifs, visuels et humains montrant un passage de 60 % à presque 100 % de réussite sans forte perte esthétique.

---

## 2. Travaux précédents

### 2.1 Modèles de diffusion d'images

Un modèle de diffusion apprend à reconstruire des images après qu'on leur a ajouté du bruit. Pour générer une nouvelle image, on part donc d'un bruit presque aléatoire, puis on le nettoie étape après étape.

Des chercheurs ont proposé le **Classifier Guidance**, qui utilise le gradient d'un classificateur pour orienter cette reconstruction vers une catégorie donnée. D'autres travaux ont ensuite généralisé l'idée pour guider plus librement le modèle.

Les images en haute résolution demandent beaucoup de calcul. Les **Latent Diffusion Models** réduisent ce coût en compressant l'image avec un VAE. La diffusion se fait alors dans un espace latent plus petit, tout en conservant une bonne qualité visuelle. Ces progrès ont permis des outils d'édition d'image et de génération texte-vers-image comme DALL-E 2 ou Midjourney.

### 2.2 QR codes esthétiques

#### Méthodes non génératives

Les premiers travaux reposent surtout sur trois stratégies : déformer les modules, réorganiser certains modules ou transférer le style d'une image.

- Les QR codes en demi-teinte intègrent une image en changeant la forme et la taille des modules.
- **Qart** réorganise certains modules grâce aux libertés offertes par le codage et la correction d'erreur.
- D'autres méthodes utilisent la zone d'intérêt, l'attention visuelle ou les niveaux de gris pour améliorer l'apparence.
- SEE QR Code, ArtCoder et MDCM cherchent à réduire les défauts visuels du transfert de style.

Ces approches ont souvent besoin d'une image de référence, ce qui limite la variété et la liberté de création.

#### Méthodes génératives

QR Diffusion, QR Code AI Art et QRBTF utilisent des modèles de diffusion, souvent guidés par ControlNet. Toutefois, leur guidage ne tient pas toujours compte assez précisément du fonctionnement interne d'un QR code.

Text2QR suit un processus en trois étapes : il produit d'abord un QR code artistique illisible, puis applique une correction séparée. Selon les auteurs, sa phase de diffusion ne garantit pas elle-même la lisibilité. DiffQRCoder cherche au contraire à intégrer le critère de lecture directement dans le processus de diffusion, sans entraîner un nouveau modèle.

---

## 3. Méthode proposée

### 3.0 Vue d'ensemble

Le système reçoit :

- un QR code cible, noté **y** ;
- un texte descriptif, noté **p**.

Le QR code est une grille de modules. Chaque module est lui-même représenté par plusieurs pixels dans l'image.

Le processus comporte deux étapes :

1. **Étape 1 - créer l'apparence.** ControlNet produit une image artistique. Elle est jolie, mais pas forcément scannable. Cette image est notée **x̂**.
2. **Étape 2 - restaurer la lisibilité.** Le système remet du bruit sur cette image et recommence une diffusion guidée. Cette fois, il utilise à la fois une mesure de lisibilité du QR code et une mesure de ressemblance visuelle avec x̂.

Une dernière correction SR-MPGD peut encore être appliquée.

### Analogie simple

Imagine un élève qui doit recopier un texte en réalisant une calligraphie artistique. Au premier essai, le dessin est très beau mais certaines lettres sont méconnaissables. Au second essai, un correcteur lui indique uniquement les lettres illisibles. L'élève les répare en essayant de conserver son style. Le post-traitement correspond à une dernière relecture.

### 3.1 Scanning-Robust Loss (SRL)

La **SRL** est la fonction qui mesure si l'image respecte assez bien le QR code cible. Elle est adaptée au fonctionnement réel d'un scanner.

#### a. Conversion en niveaux de gris

Le scanner s'intéresse surtout à la luminosité, pas à la couleur exacte. L'image rouge-vert-bleu est donc convertie en niveaux de gris :

`G(x) = 0,299 × rouge + 0,587 × vert + 0,114 × bleu`

Le vert pèse davantage parce que la perception humaine et les normes vidéo ne donnent pas la même importance aux trois couleurs.

#### b. Matrice d'erreur pixel par pixel

Le système compare la luminosité de l'image avec la couleur attendue dans le QR code. Si une zone censée être blanche est trop sombre, ou si une zone censée être noire est trop claire, l'erreur augmente.

L'équation 1 sépare donc deux cas : l'erreur des modules blancs et celle des modules noirs. Le symbole de multiplication terme à terme signifie que chaque pixel est traité à sa position.

#### c. Importance plus forte du centre

Tous les pixels d'un module n'ont pas la même importance pour un scanner. Les pixels proches du centre ont davantage de chances de déterminer si le module est lu comme noir ou blanc.

La méthode applique une pondération gaussienne : le centre reçoit un poids fort, les bords un poids plus faible. L'équation 2 additionne les erreurs pondérées à l'intérieur de chaque module.

#### d. Sous-module central

Chaque module est imaginé comme une grille de 3 × 3 sous-zones. La méthode regarde spécialement la zone centrale. Elle calcule sa luminosité moyenne, puis la transforme en décision binaire :

- moyenne inférieure à 0,5 : plutôt noir ;
- moyenne supérieure ou égale à 0,5 : plutôt blanc.

Si cette décision correspond au module cible, le module est considéré comme correctement décodable.

#### e. Arrêt anticipé

Une fois qu'un module est correct, l'algorithme cesse de le modifier. Cela évite de « trop corriger » une zone déjà lisible et de détériorer inutilement l'image.

L'équation 6 calcule finalement la moyenne des erreurs des modules encore incorrects. C'est la perte SRL.

### Ce qu'il faut retenir des équations 1 à 6

Elles traduisent une idée assez simple : **corriger surtout le centre des carrés mal lus, et laisser tranquilles ceux qui sont déjà corrects**.

---

### 3.2 Pipeline en deux étapes et guidage SRPG

#### Étape 1

Le texte et le QR code sont convertis en représentations numériques. Le modèle part d'un bruit aléatoire et ControlNet crée progressivement une image correspondant au prompt. Le résultat x̂ sert de référence esthétique pour la suite.

#### Étape 2

L'image x̂ est encodée par le VAE dans l'espace latent, puis on lui ajoute du bruit. Le QR code cible est aussi transformé avec Qart pour ressembler davantage au motif de l'image artistique. Ce QR code adapté est noté **ỹ**.

À chaque étape de débruitage, le système estime ce que serait l'image propre, puis la décode temporairement afin de mesurer deux erreurs :

- **L_SR** : erreur de lecture du QR code ;
- **L_LPIPS** : différence perceptuelle avec la belle image produite à l'étape 1.

La fonction de guidage est :

`F_SRP = λ1 × L_SR + λ2 × L_LPIPS`

Les nombres **λ1** et **λ2** règlent le compromis :

- augmenter λ1 donne plus d'importance à la lecture ;
- augmenter λ2 donne plus d'importance au maintien de l'apparence.

Le gradient de cette fonction modifie la prédiction du bruit. Après environ 40 itérations dans les expériences, le latent final est décodé en QR code artistique.

### Ce que signifient les équations 7 à 11

Elles décrivent mathématiquement la boucle suivante : prédire l'image propre, mesurer les deux erreurs, calculer dans quelle direction corriger le latent, puis effectuer une étape de débruitage.

---

### 3.3 Post-traitement SR-MPGD

Même après la deuxième étape, certains modules peuvent rester fragiles. SR-MPGD cherche alors à diminuer encore l'erreur de lecture, mais impose que le résultat reste dans l'ensemble des images naturelles que le VAE sait représenter.

À chaque itération :

1. le VAE décode le latent en image ;
2. la méthode calcule la perte SRL et la perte LPIPS ;
3. elle modifie légèrement le latent dans la direction qui réduit ces pertes ;
4. le VAE ramène implicitement le résultat vers une image plausible.

La perte totale du post-traitement est :

`L = L_SR + λ × L_LPIPS`

LPIPS empêche les corrections techniques de transformer brutalement le contenu général de l'image. Les équations 12 à 14 formalisent cette descente de gradient.

---

## 4. Expériences

### 4.1 Réglages expérimentaux

Les auteurs utilisent 100 prompts générés par GPT-4. Stable Diffusion emploie le modèle Cetus-Mix Whalefall et ControlNet le modèle QR Code Monster v2, avec une force de guidage de 1,35.

Les comparaisons portent sur QR Code AI Art, QR Diffusion et QRBTF. Les méthodes non génératives ne sont pas incluses parce qu'elles demandent une image esthétique de référence, alors que cette expérience utilise seulement du texte. Certaines autres méthodes ne sont pas disponibles en code ouvert.

La plupart des tests emploient :

- un QR code de version 3 ;
- une correction d'erreur moyenne M, soit 15 % ;
- le motif de masque numéro 4 ;
- une bordure de 80 pixels ;
- des modules de 20 × 20 pixels ;
- le message « Thanks reviewer! ».

Les calculs sont réalisés sur une NVIDIA RTX 4090. Une génération complète prend environ 14 à 18 secondes, avec 40 étapes d'inférence dans chacune des deux phases.

Trois mesures sont utilisées :

- **SSR** : taux de lecture par qr-verify ;
- **CLIP-aes.** : estimation automatique de la qualité esthétique ;
- **CLIP-score** : correspondance entre l'image et le texte demandé.

### 4.2 Comparaison quantitative

| Méthode | SSR | CLIP-aes. | CLIP-score |
|---|---:|---:|---:|
| QR Code AI Art | 90 % | 5,7003 | 0,2341 |
| QR Diffusion | 96 % | 5,5150 | 0,2780 |
| QRBTF | 56 % | **7,0156** | **0,3033** |
| DiffQRCoder | **99 %** | 6,8233 | 0,2992 |

### Interprétation

QRBTF obtient la meilleure note esthétique, mais presque un code sur deux échoue au scan dans ce test. QR Diffusion est assez robuste, mais sa note esthétique est plus faible. DiffQRCoder cherche le compromis : il n'est pas premier sur chaque mesure prise séparément, mais il combine un taux de lecture très élevé avec une bonne apparence.

Attention : CLIP-aes. et CLIP-score sont des évaluations automatiques, pas une vérité absolue sur la beauté d'une image.

### Robustesse à l'angle

| Rotation simulée | 0° | 15° | 30° | 45° |
|---|---:|---:|---:|---:|
| SSR | 100 % | 100 % | 100 % | 97 % |

Les codes testés sont tournés avec CSS. Un code est déclaré lisible s'il est scanné en moins de trois secondes.

### Niveaux de correction d'erreur

| Niveau | L (7 %) | M (15 %) | Q (25 %) | H (30 %) |
|---|---:|---:|---:|---:|
| SSR | 96 % | 100 % | 100 % | 100 % |

Un niveau de correction élevé permet au QR code de survivre à davantage d'erreurs. Le résultat intéressant est donc le score de 96 % au niveau L, le plus exigeant dans ce contexte parce qu'il tolère le moins de dégâts.

### Différents scanners

| Scanner | qr-verify | iPhone 13 | Pixel 7 |
|---|---:|---:|---:|
| SSR | 100 % | 97 % | 88 % |

Pour ce test, 30 QR codes sont scannés dix fois avec chaque lecteur. Le Pixel 7 obtient le résultat le plus faible, ce qui montre que la réussite dépend aussi du logiciel et de l'appareil utilisés.

### Différents messages encodés

| Message | SSR |
|---|---:|
| « Je pense, donc je suis. » | 97 % |
| « Tu es la prunelle de mes yeux. » | 100 % |
| https://www.google.com.tw/ | 100 % |
| https://www.wikipedia.org/ | 97 % |

La structure d'un QR code change selon le message. Ces tests vérifient que la méthode ne fonctionne pas uniquement avec une seule grille.

### Comparaison visuelle et évolution des erreurs

Les auteurs estiment que DiffQRCoder mélange plus harmonieusement le motif du code avec le contenu demandé que QR Code AI Art et QR Diffusion. Par rapport à QRBTF, il sacrifie un peu d'esthétique pour gagner nettement en lisibilité.

La figure 6 montre les modules incorrects en rouge pendant le débruitage. Ils diminuent progressivement. Lorsque leur quantité passe sous ce que la correction d'erreur du QR code peut supporter, le code devient scannable.

### Étude auprès de 387 personnes

Les participants classent quatre QR codes selon leur apparence. Un rang moyen plus faible est meilleur.

| Méthode | Rang esthétique moyen | SSR |
|---|---:|---:|
| QR Code AI Art | 2,71 | 90 % |
| QR Diffusion | 3,18 | 96 % |
| QRBTF | **1,86** | 56 % |
| DiffQRCoder | 2,25 | **99 %** |

QRBTF est préféré visuellement, mais fonctionne beaucoup moins souvent. DiffQRCoder arrive deuxième pour l'apparence et premier pour la lecture. L'étude a reçu une autorisation éthique ; les participants ont donné leur consentement et les auteurs déclarent ne pas avoir collecté de données personnelles sensibles.

### 4.3 Étude d'ablation

Une étude d'ablation retire ou modifie un composant pour savoir à quoi il sert réellement.

Avec ControlNet seul, le score esthétique est de 7,0661 et le SSR de 60 %. Lorsque la force λ1 de la correction QR augmente, le SSR monte généralement, tandis que l'esthétique baisse légèrement. Avec λ1 = 600 sans post-traitement, le SSR atteint 94 %.

Le post-traitement SR-MPGD augmente encore le taux de lecture. Par exemple, avec λ1 = 500 et sans terme LPIPS pendant la deuxième étape, le SSR passe de 88 % à 100 %. Dans la configuration principale λ1 = 500 et λ2 = 3, il passe de 89 % à 99 %, avec une variation esthétique très faible.

Augmenter λ2 aide à conserver l'apparence de la première image, même si la relation n'est pas parfaitement monotone dans le petit tableau présenté. Cela illustre le compromis général entre liberté artistique et contraintes du QR code.

---

## 5. Conclusion traduite

Les auteurs présentent DiffQRCoder, un générateur de QR codes par diffusion qui ne nécessite pas de nouvel entraînement. Ils créent une perte SRL destinée à améliorer la lecture, puis l'intègrent au guidage SRPG d'un pipeline en deux étapes. Ils ajoutent enfin SR-MPGD pour renforcer la fiabilité.

Selon leurs expériences, la méthode augmente fortement le taux de lecture par rapport aux approches existantes, sans perte importante d'attrait visuel, et semble adaptée à des applications réelles.

---

# Annexes traduites et expliquées

## A. Conversion en niveaux de gris

L'annexe précise les coefficients utilisés : 0,299 pour le rouge, 0,587 pour le vert et 0,114 pour le bleu. Ils correspondent à une conversion normalisée de l'espace colorimétrique YCbCr.

## B. Détails du SRPG

### B.1 LPIPS

Comparer directement deux images pixel par pixel donne parfois un mauvais résultat : un léger déplacement peut créer une grande erreur numérique alors que les images paraissent presque identiques à l'œil humain.

LPIPS utilise des réseaux déjà entraînés, comme VGG ou AlexNet, pour extraire des caractéristiques visuelles à plusieurs niveaux. La distance est calculée entre ces caractéristiques. Dans DiffQRCoder, LPIPS aide donc à conserver le sujet, la composition et l'apparence globale de l'image de l'étape 1.

### B.2 Guidage conditionnel

Les équations 15 et 16 relient la prédiction de bruit du modèle à une probabilité conditionnelle. En clair, on ne demande plus seulement : « quel bruit faut-il retirer pour obtenir une image plausible ? », mais aussi : « quel bruit faut-il retirer pour obtenir une image plausible qui respecte ce QR code ? »

Le théorème de Bayes permet de séparer la plausibilité générale de l'image et le respect de la condition. Le gradient de la fonction de guidage fournit la correction supplémentaire.

### B.3 Gradient du SRPG

L'équation 17 reconstruit une estimation de l'image propre à partir du latent bruité. L'équation 18 applique la règle de dérivation en chaîne à travers le décodeur du VAE et le prédicteur de bruit.

Idée simple : comme l'erreur est mesurée sur l'image finale mais que les corrections se font dans l'espace latent, le système doit calculer comment une petite modification du latent changera l'image, puis comment ce changement modifiera les deux pertes.

## C. Algorithme complet

Qart exploite les degrés de liberté du codage QR pour adapter le motif cible à l'image artistique sans changer le message décodé.

L'algorithme complet peut se résumer ainsi :

1. tirer un bruit aléatoire ;
2. effectuer la première diffusion avec ControlNet ;
3. décoder la belle image x̂ ;
4. adapter le QR code cible avec Qart ;
5. encoder x̂ et lui ajouter du bruit ;
6. recommencer la diffusion ;
7. à chaque étape, estimer l'image propre et mesurer son taux d'erreur ;
8. si l'erreur dépasse ce que le QR code peut corriger, activer SRPG ;
9. sinon, poursuivre sans correction supplémentaire ;
10. décoder et renvoyer l'image finale.

Ce seuil évite de continuer à forcer la grille lorsque le code est déjà suffisamment fiable.

## D. Détails des expériences

Les auteurs utilisent principalement la bibliothèque `diffusers` de Hugging Face. Ils publient les réglages des services comparés : taille et marge pour QRBTF, poids ControlNet et force de diffusion pour QR Code AI Art, poids du code pour QR Diffusion, et force 1,35 pour QR Code Monster.

Pour mesurer l'erreur d'un QR code, ils comptent la proportion de modules dont la zone centrale serait mal décodée. Les graphiques montrent que SRPG réduit fortement l'erreur durant les cinq premières itérations. Sans ce guidage, l'amélioration est plus lente. La force du gradient diminue ensuite au cours des étapes, signe que les corrections deviennent moins nécessaires lorsque l'image se rapproche de l'objectif.

## E. Étude utilisateur

Les participants voient quatre images, une par méthode, puis les classent de la plus à la moins esthétique. Le rang moyen est une moyenne pondérée. Par exemple, si 20 personnes donnent le rang 1, 10 le rang 2 et 100 le rang 3, le rang moyen vaut :

`(1 × 20 + 2 × 10 + 3 × 100) / 130 = 2,615`

## F. Limites et travaux futurs

La méthode ne garantit pas toujours 100 % de réussite et demande d'ajuster des hyperparamètres. Les auteurs utilisent donc un post-traitement. Ils souhaitent à l'avenir créer un système complet de bout en bout, moins sensible aux réglages et n'ayant plus besoin de cette dernière correction. Ils envisagent aussi des méthodes image-vers-image pour permettre une personnalisation plus précise.

## G. Effets possibles sur la société

Des QR codes artistiques peuvent être utilisés pour l'hameçonnage, le spam ou la diffusion de contenus faux ou inappropriés. Les auteurs suggèrent de filtrer les URL et de bloquer certains prompts.

---

# Lecture critique : ce que l'article prouve, et ce qu'il ne prouve pas

## Points forts

- Le problème étudié est concret : un QR code artistique inutile à scanner ne sert à rien.
- La fonction SRL tient compte du fonctionnement des modules et de l'importance de leur centre.
- Les auteurs testent plusieurs angles, niveaux de correction, messages et appareils.
- L'étude d'ablation montre que les nouvelles étapes améliorent réellement le taux de lecture.
- La méthode réutilise des modèles existants sans nouvel entraînement coûteux.

## Précautions

- Les résultats portent sur 100 prompts et des réglages précis ; ils ne garantissent pas le même score pour toutes les images possibles.
- Les appareils n'obtiennent pas tous le même résultat : le Pixel 7 descend à 88 % dans le test correspondant.
- Le meilleur score esthétique automatique n'appartient pas à DiffQRCoder.
- La beauté est subjective. L'étude humaine aide, mais elle ne suffit pas à définir un goût universel.
- Les rotations sont simulées à l'écran avec CSS ; cela ne reproduit pas toutes les difficultés du monde réel, comme les reflets, une mauvaise impression, le flou ou un éclairage faible.
- Le temps annoncé de 14 à 18 secondes utilise une RTX 4090, une carte graphique très puissante.
- Le post-traitement et les réglages manuels montrent que le système n'est pas encore entièrement automatique.

## Conclusion pour un lycéen

Ce travail est un bon exemple d'ingénierie de l'IA : les chercheurs n'inventent pas forcément un énorme nouveau modèle. Ils prennent une IA existante et lui ajoutent une règle précise fondée sur la structure du problème.

La grande idée est de ne pas obliger chaque pixel à être exactement noir ou blanc. Le scanner a surtout besoin que certaines zones centrales restent interprétables et que le nombre d'erreurs reste sous la capacité de correction du QR code. Les pixels moins importants peuvent alors former des arbres, des fenêtres, de l'eau ou des lumières. C'est cette liberté contrôlée qui permet de réunir fonction technique et création artistique.

---

## Une phrase pour retenir tout l'article

**DiffQRCoder laisse l'IA dessiner librement, puis corrige seulement ce qui empêche le QR code d'être lu, tout en protégeant l'apparence générale de l'image.**
