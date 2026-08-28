# E035 — récupération du parent Stage 2

## Fait établi

L’archive E034 contient les PNG, traces et empreintes, mais pas le tenseur latent Stage 2.
Un PNG ne permet pas de reconstruire exactement ce latent : un nouvel encodage VAE
créerait un autre parent. E035 distingue donc strictement deux situations.

## Voie A — latent E034 encore disponible

Cette voie est la seule qui permette de reprendre exactement le parent Stage 2 d’E034.
Le JSON source doit conserver les empreintes du Stage 1 archivé et
`generation.stage1_regenerated=false`.

```bash
python scripts/export_e035_parent_artifact.py   --image /chemin/parent-stage2.png   --latent /chemin/parent-stage2-latent.safetensors   --source-json docs/e035-parent-source-template.json   --output-dir /data/e035-parent-v1

python -m prooftag_qr.e035_parent_artifact /data/e035-parent-v1
```

Les fichiers pickle (`.pt`, `.pth`) sont refusés. Le latent doit déjà être un
`safetensors` de provenance contrôlée.

## Voie B — parent apparié depuis le Stage 1 exact d’E034

Le bundle inclut byte pour byte le raster Stage 1 observé dans E034 :

```text
docs/e035-assets/e034-observed-stage1.png
file SHA-256  : be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71
raster SHA-256: ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079
```

Lorsque le latent E034 n’existe plus, le Job séparé `capture-parent` :

1. vérifie ces deux empreintes ;
2. **ne régénère pas Stage 1** ;
3. rejoue uniquement le Stage 2 E033 figé ;
4. exporte immédiatement le PNG et le latent en safetensors ;
5. écrit un audit et un contrat immuable.

```bash
bash scripts/deploy-e035-notebook.sh prepare
bash scripts/deploy-e035-notebook.sh capture-parent
bash scripts/deploy-e035-notebook.sh verify-parent
```

Ce parent est valable pour comparer les deux losses, car les branches partent du même
latent vérifié. Il ne doit toutefois pas être présenté comme le latent Stage 2 exact
observé dans E034. Sa provenance est
`stage2_replayed_from_exact_e034_stage1`.

## Contrat fail-closed

Le répertoire `/data/e035-parent-v1` doit contenir :

```text
parent-stage2.png
parent-stage2-latent.safetensors
parent-stage2-metadata.json
```

Le chargement échoue sur fichier absent, nom non canonique, hash fichier ou raster
différent, latent altéré/non fini, révision différente, Stage 1 non conforme ou
`stage1_regenerated` différent de `false`.
