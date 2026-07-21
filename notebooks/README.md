# Notebook d'autopsie SRPG

Le notebook `01_srpg_step_by_step.ipynb` ouvre directement une archive de benchmark, affiche la
chaîne brute → SRPG → finale, trace les 40 diagnostics et, pour les campagnes récentes, montre les
estimations `x0` et cartes d'erreurs enregistrées pendant le débruitage.

## Installation et lancement

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
python -m pip install -e ".[notebook]"
$env:PROOFTAG_QR_BENCHMARK_ARCHIVE = "$HOME\Downloads\prooftag-benchmarks\20260721T090541Z-0b3c040b.tar.gz"
jupyter lab notebooks\01_srpg_step_by_step.ipynb
```

Sans variable d'environnement, le notebook sélectionne la dernière archive trouvée dans
`Downloads/prooftag-benchmarks`. Modifier `CASE` dans la deuxième cellule pour examiner un autre
cas.

Le notebook réutilise d'abord le dossier déjà extrait à côté de l'archive. S'il doit extraire
lui-même l'archive, il écrit dans `.prooftag-notebook-cache` au même emplacement, jamais dans le
dossier `notebooks`. Un emplacement différent peut être imposé avec
`PROOFTAG_QR_NOTEBOOK_CACHE`.

Les archives antérieures à l'instrumentation montrent `raw`, `srpg`, `final` et les courbes, mais
pas les images intermédiaires. Relancer `make benchmark-e005` après déploiement du code récent pour
obtenir `srpg_control` et les checkpoints `srpg_step_XX_x0/errors`.
