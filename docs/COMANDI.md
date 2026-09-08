# Comandi — come far girare questo repository

> Aggiornato: **2026-09-01**, dopo i fix descritti in
> [`INTEGRAZIONE_REPORT.md`](./INTEGRAZIONE_REPORT.md).
>
> Riferimento pratico: ogni comando, cosa fa, quanto ci mette, cosa scrive su
> disco. I tempi sono misurati su questa macchina (Quadro P2200, 5 GB, EuroSAT
> locale) e vanno presi come ordini di grandezza.

---

## 0. Tre regole prima di lanciare qualsiasi cosa

**1. Lanciare sempre dalla radice del repository**, mai da dentro `src/`:

```bash
cd C:/Users/info/Desktop/Repos/APAI_exam
python src/coop.py            # SÌ
```

```bash
cd src && python coop.py      # NO
```

Non è pedanteria: alcuni script risolvono i percorsi rispetto alla directory da
cui li lanci. Prima dei fix, lanciare il driver di Marco da `src/` creava una
seconda cartella `src/plots/` — che infatti è finita tracciata nel repository.
Oggi i percorsi dei plot sono assoluti, derivati da `__file__`, ma
`get_dataloaders(root="./data")` usa ancora `./data`: lanciato da `src/`
ri-scaricherebbe EuroSAT in `src/data/`.

**2. La prima esecuzione di un modello scarica i pesi.** OpenCLIP prende
`ViT-B-32` / `laion2b_s34b_b79k` da HuggingFace (~600 MB) e lo mette nella
cache utente. Succede una volta sola; i comandi qui sotto assumono la cache già
popolata. Il warning `You are sending unauthenticated requests to the HF Hub`
è normale e si può ignorare.

**3. Nessuno di questi comandi fa commit.** Gli unici comandi git elencati in
questo documento sono di sola lettura o di ripristino locale, tranne i due
della sezione 6.3, che sono segnalati esplicitamente.

---

## 1. Smoke test — "il codice funziona?"

Questi comandi non addestrano niente e non producono risultati per il report.
Servono a rispondere a una domanda sola: **l'ambiente e il codice sono sani?**
Lanciali in quest'ordine dopo ogni `git pull`, ogni merge, o ogni volta che
qualcosa si comporta in modo strano.

| Comando | Cosa verifica | Tempo |
|---|---|---|
| `python src/base_model.py` | Backbone congelato (0 parametri addestrabili), forme delle feature, il `predict()` condiviso, la costruzione dei prompt per due dataset diversi | ~20 s |
| `python src/dataset.py` | I dataloader restituiscono `(images, labels, text_descriptions)` per EuroSAT, DTD e Flowers102 | ~1 min, **scarica DTD e Flowers102 la prima volta** (lenta) |
| `python src/few_shot.py` | Campionamento K-shot bilanciato, senza decodificare le immagini | ~15 s |
| `python src/coop.py` | CoOp: conteggio parametri per M=4/8/16, unified vs CSC, e che il gradiente arrivi a `ctx` | ~40 s |
| `python src/clip_adapter.py` | CLIP-Adapter, Tip-Adapter e LoRA: forme, parametri, alpha fisso vs appreso, `predict()` ereditato | ~90 s |
| `python src/coop_adapter.py` | Modello congiunto: che l'ereditarietà multipla **fallisca** come previsto, e che quello esplicito funzioni con gradienti a entrambi i moduli | ~90 s |
| `python src/baselines.py` | Zero-Shot e Linear Probe (che viene addestrato su tutto EuroSAT) | ~5 min |
| `python src/optimal_transport.py` | Sinkhorn su un batch fittizio, forme della matrice delle distanze | ~30 s |

### Cosa devi vedere

`python src/base_model.py`:

```
Trainable params    : 0
Default prompts     : ['a satellite image of annual crop land', ...]
predict() shapes    : preds (2,), sims (2, 10)
DTD prompts         : ['a photo of a banded texture', ...]
```

`python src/coop.py` — il numero che conta è l'ultimo:

```
ctx grad norm: 1.8757e+03  (must be > 0)
```

Se fosse `0` o desse `AttributeError: 'NoneType' has no attribute 'norm'`,
significa che qualcuno ha aggiunto un `@torch.no_grad()` a un metodo override e
**CoOp sta girando senza imparare nulla, in silenzio**.

`python src/clip_adapter.py` deve stampare **tre** blocchi:

```
Testing CLIPAdapterModel ...
predict() inherited from: BaseCLIPWrapper
Learnable alpha: start 0.2000 | trainable 131,713 (fixed-alpha variant: 131,712)
Testing TipAdapterModel ...
Testing VisionLoRAModel ...
```

Se ne stampa **solo il primo**, il working tree è tornato a una versione
precedente al lavoro di Marco. Vedi la sezione 7.

`python src/coop_adapter.py` — la prima riga deve essere un fallimento atteso:

```
[ok] naive multiple inheritance fails as expected: '_Naive' object has no attribute 'prompt_learner'
M=16  unified  r=4   | context   8,192 + adapter 131,712 = 139,904
ctx grad norm     : 4.3644e+03  (must be > 0)
adapter grad norm : 2.4757e+02  (must be > 0)
```

---

## 2. Esperimenti di Carlo — CoOp

Tutto passa da `src/coop_experiments.py`. I risultati si accumulano in
`plots/coop_results.json`, indicizzati per **tag**; i grafici vanno in
`plots/coop_sweeps.png` e `plots/coop_training_curves.png`.

### 2.1 Rigenerare i grafici senza riaddestrare

```bash
python src/coop_experiments.py --plot
```

Legge `plots/coop_results.json` e ridisegna le due figure. **Non addestra
niente**, ci mette pochi secondi. È il comando da usare ogni volta che si
cambia qualcosa nel codice dei plot.

### 2.2 Le quattro ablation del brief

```bash
python src/coop_experiments.py --sweep lr        # 2e-6 … 2e-1, sei run
python src/coop_experiments.py --sweep ctx_len   # M = 4, 8, 16
python src/coop_experiments.py --sweep csc       # unified vs class-specific
python src/coop_experiments.py --sweep shots     # K = 1, 2, 4, 8, 16
python src/coop_experiments.py --sweep all       # tutte e quattro di seguito
```

⚠️ **Il default è `--epochs 50`, ma i risultati già nel JSON sono a 100
epoche.** Per riprodurli o estenderli serve:

```bash
python src/coop_experiments.py --sweep lr --epochs 100
```

Il tag contiene le epoche (`CoOp_M16_unified_K16_lr0.002_e100`), quindi una run
a 50 epoche **non** sovrascrive quella a 100 — ma finisce comunque nel JSON e
nei grafici accanto a lei, il che confonde. È già successo.

Tempo: circa 2–4 minuti a run a 100 epoche, di cui ~45 s sono la valutazione
sull'intero test set. Uno sweep `lr` completo sta sui 15–20 minuti.

### 2.3 Una singola configurazione

```bash
python src/coop_experiments.py --single --n-ctx 16 --shots 16 --lr 2e-3 --epochs 100
python src/coop_experiments.py --single --csc --n-ctx 16 --epochs 100
python src/coop_experiments.py --single --ctx-init "a satellite image of" --n-ctx 4 --epochs 100
```

| Flag | Default | Cosa fa |
|---|---|---|
| `--n-ctx` | 16 | Numero di token di contesto `M` |
| `--shots` | 16 | Immagini etichettate per classe |
| `--csc` | off | Class-Specific Context (un prompt per classe) invece di Unified |
| `--ctx-init` | nessuno | Inizializza il contesto da una frase invece che da rumore. **Deve tokenizzare a esattamente `--n-ctx` token**, altrimenti alza un `ValueError` esplicito |
| `--lr` | 2e-3 | Learning rate (SGD + momentum 0.9) |
| `--epochs` | 50 | Epoche |
| `--batch-size` | 32 | Con 160 immagini dà 5 step per epoca |
| `--seed` | 42 | Seed del campionamento K-shot |
| `--val-subset` | 1000 | Immagini usate per la validazione **durante** il training. La valutazione finale è sempre su tutto il test set |
| `--num-workers` | 0 | Su Windows lasciare 0 |
| `--wandb` | off | Logga su Weights & Biases |

### 2.4 Prova rapida che tutto gira

```bash
python src/coop_experiments.py --single --n-ctx 4 --shots 4 --epochs 3 --val-subset 400
```

~1 minuto. **Attenzione**: aggiunge una run vera al JSON, con tag
`CoOp_M4_unified_K4_lr0.002_e3`. Se era solo una prova, va tolta a mano dal
file prima di rigenerare i grafici.

---

## 3. Esperimenti di Marco — CLIP-Adapter

```bash
python src/clip_adapter_experiments.py --epochs 5 --batch_size 64 --lr 1e-3 --no_wandb
```

Fa due sweep di seguito: **reduction ratio** (r = 16, 8, 4, 2) e **alpha**
(0.1, 0.2, 0.5, 0.8). Scrive `plots/clip_adapter_results.json`,
`plots/clip_adapter_alpha_results.json` e le due figure corrispondenti.

| Flag | Default | Note |
|---|---|---|
| `--epochs` | 5 | Attenzione: gira **full-shot** su 21.600 immagini |
| `--batch_size` | 64 | |
| `--lr` | 1e-3 | AdamW |
| `--no_wandb` | off | Da usare sempre, se non si vuole loggare su W&B |

⚠️ **Questo comando è lungo.** Una singola epoca full-shot costa ~220 s di
training + ~45 s di valutazione. Con 8 configurazioni e 5 epoche si sta oltre
le **due ore**. Per una verifica veloce, `--epochs 1` su una configurazione
sola si fa da Python (sezione 4.4).

### Alpha: due esperimenti, non uno

Dopo i fix, `alpha` è **fissa per default**. Questo era necessario perché uno
sweep su un parametro che poi si muove da solo non misura niente.

```python
from src.clip_adapter_experiments import run_single_experiment

# alpha fisso: è lo sweep del brief
run_single_experiment(reduction_ratio=4, alpha=0.2, epochs=5)

# alpha appreso e vincolato in (0,1): esperimento distinto
run_single_experiment(reduction_ratio=4, alpha=0.2, learnable_alpha=True, epochs=5)
```

La modalità finisce nel tag della run (`..._fixed` / `..._learned`), così una
non sovrascrive l'altra, e il valore a cui alpha si assesta viene registrato
come `alpha_final` nei risultati.

---

## 3-bis. La griglia comparativa — `src/run_all.py`

È **il** driver degli esperimenti: addestra e valuta ogni metodo su ogni
dataset e produce tutte le figure e le tabelle che finiscono nel report.
Dopo i fix, `plots/` è scritta da questo script e da nessun altro
(`engine.py` lanciato da solo scrive in `plots/baselines_only/`), così la
cartella non può più contenere due generazioni di figure che si
contraddicono.

```bash
# Griglia principale: 16-shot, budget appaiato, tre semi.
python src/run_all.py --shots 16 --seeds 0 1 2 --epochs 10

# Riferimento full-shot, un solo seme.
python src/run_all.py --shots 0 --seeds 0 --epochs 10 --out fullshot_results.json

# Un dataset alla volta: un'interruzione costa un dataset, non tutti e tre.
python src/run_all.py --dataset eurosat --shots 16 --seeds 0 1 2

# Prova end-to-end veloce, pochi minuti.
python src/run_all.py --dataset eurosat --shots 16 --seeds 0 --epochs 2 --no-tsne

# Solo le figure, dai risultati già su disco. Nessuna GPU.
python src/run_all.py --plots-only
```

| Flag | Default | Note |
|---|---|---|
| `--shots K` | 16 | Immagini etichettate per classe, **le stesse per ogni metodo**. `0` = full-shot |
| `--seeds` | `0` | Uno o più semi; i risultati escono come media ± deviazione standard |
| `--epochs` | 10 | **Uguale per tutti** i metodi addestrabili: è ciò che rende il confronto un confronto |
| `--dataset` | tutti | `eurosat`, `dtd`, `flowers102` |
| `--adapter-lr-scale` | 1.0 | Moltiplicatore del learning rate per la metà "adapter" del modello congiunto |
| `--skip-slow` | off | Salta l'Optimal Transport sopra le 10 classi (su Flowers102 costa ~5 minuti a passata) |
| `--no-tsne` | off | |
| `--out` | `unified_results.json` | Nome del JSON dentro `plots/` |
| `--plots-only` | off | Ridisegna tutto senza toccare la GPU |

### Cosa è cambiato, e perché

Tre cose che nella prima versione rendevano i numeri non confrontabili:

- **Il budget.** CoOp girava 10 epoche con SGD, l'adapter e LoRA 5 con AdamW.
  Dire "l'adapter batte CoOp di 1,2 punti" confondeva il metodo con il
  budget. Ora `--epochs` vale per tutti; l'ottimizzatore resta quello del
  paper di ciascun metodo, perché quello *fa parte* del metodo.
- **La cache di Tip-Adapter.** Era costruita sull'intero training split
  (`num_shots=99999`): 21.600 chiavi, e Tip-Adapter-F dichiarava 11.059.200
  parametri addestrabili. Ora la cache è K-shot e viene dallo **stesso**
  support set seedato che vedono gli altri metodi, quindi non dipende più
  nemmeno dall'ordine di shuffle del loader.
- **Il modello congiunto.** Veniva addestrato con la ricetta di CoOp (SGD
  2e-3) applicata anche alle 131.712 righe dell'adapter, che vuole AdamW
  1e-3: l'adapter non si muoveva e il congiunto perdeva contro l'adapter da
  solo su tutti e tre i dataset. Ora ci sono due gruppi di ottimizzazione
  (`CoOpAdapterModel.trainable_param_groups`) sotto AdamW.

### Le due tabelle

Lo script stampa **due** tabelle markdown, non una: accuratezza e parametri
addestrabili. Non è pignoleria — la versione precedente aveva una sola
colonna `Params`, riempita dentro il ciclo sui dataset, quindi sopravviveva
solo il valore dell'ultimo: con l'ordine EuroSAT, DTD, Flowers102 ogni riga
mostrava i parametri di Flowers102, e il Linear Probe risultava 52.326 sulla
riga EuroSAT dove sono 5.130.

### Costo

Sulla macchina di Marco una passata full-shot su tre dataset e dieci metodi
ha richiesto **167 minuti**. Il regime 16-shot è molto più economico —
160 immagini per epoca su EuroSAT invece di 21.600 — ma la valutazione gira
sempre sull'intero test set, quindi il costo per seme non scende
proporzionalmente. Con tre semi, calcolare il tempo su un dataset solo prima
di lanciare tutto.

---

## 4. Uso da Python — le API

Le cose che non hanno una riga di comando dedicata.

### 4.1 Il `predict()` condiviso

Non serve più scriverlo. Ogni sottoclasse di `BaseCLIPWrapper` lo eredita:

```python
preds, scores = model.predict(images)     # (B,) e (B, num_classes)
```

Lo sovrascrivono solo `TipAdapterModel` e `OptimalTransportCLIP`, che hanno una
regola di scoring diversa. Per controllare da dove arriva:

```python
next(k.__name__ for k in type(model).__mro__ if "predict" in k.__dict__)
# 'BaseCLIPWrapper' per tutti tranne quei due
```

### 4.2 Cambiare dataset

Il template del prompt va passato insieme ai nomi delle classi, sempre. Se non
lo si fa, il modello costruisce `"a satellite image of banded"` per una texture
e l'accuratezza cala **senza nessun errore**.

```python
from src.dataset import (DTD_CLASS_NAMES, DTD_PROMPT_TEMPLATE,
                         get_dtd_dataloaders)
from src.clip_adapter import CLIPAdapterModel

model = CLIPAdapterModel(device="cuda",
                         class_names=DTD_CLASS_NAMES,
                         prompt_template=DTD_PROMPT_TEMPLATE)

model.build_prompts()[0]     # 'a photo of a banded texture'
```

`engine.train()` prende il template, in ordine di precedenza, da: argomento
esplicito → attributo del modello → default EuroSAT. Quindi con un modello
costruito bene non serve ripeterlo:

```python
train(model, train_loader, epochs=20)                       # usa il template del modello
train(model, train_loader, epochs=20,
      class_names=[...], prompt_template="a photo of a {}")  # forzato
```

CoOp e il modello congiunto sono immuni: un template non ce l'hanno, sostituirlo
è il metodo.

### 4.3 Grafici comparativi con i metodi del team

Il punto del progetto è il trade-off accuratezza / parametri addestrabili.
Perché appaia, i modelli addestrati vanno passati a `run_all_evaluations`:

```python
import torch
from src.dataset import get_dataloaders
from src.few_shot import build_few_shot_loader
from src.base_model import BaseCLIPWrapper
from src.coop import CoOpModel
from src.clip_adapter import CLIPAdapterModel
from src.coop_adapter import CoOpAdapterModel
from src.engine import train, run_all_evaluations, plot_comparative_results, plot_memory_vs_epochs

device = "cuda"
train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=0)
support = build_few_shot_loader(train_loader.dataset, n_shots=16)

# Un solo backbone congelato, riusato da tutte le baseline: su 5 GB conta.
wrapper = BaseCLIPWrapper(device=device)

coop = CoOpModel(device=device, n_ctx=16)
h_coop = train(coop, support, epochs=100, lr=2e-3, use_wandb=False, model_name="CoOp")

adapter = CLIPAdapterModel(device=device, reduction_ratio=4)
h_adapter = train(adapter, support, epochs=20, lr=1e-3, optimizer_type="adamw",
                  use_wandb=False, model_name="CLIP-Adapter")

joint = CoOpAdapterModel(device=device, n_ctx=16, reduction_ratio=4)
h_joint = train(joint, support, epochs=20, lr=1e-3, optimizer_type="adamw",
                use_wandb=False, model_name="CoOp+Adapter")

results = run_all_evaluations(
    test_loader, device=device, use_wandb=False,
    clip_wrapper=wrapper,
    train_loader=support,          # ← Linear Probe a supervisione uguale
    extra_models={"CoOp (M=16)": coop,
                  "CLIP-Adapter (r=4)": adapter,
                  "CoOp+Adapter": joint},
)

plot_comparative_results(results)
plot_memory_vs_epochs({"CoOp": h_coop, "CLIP-Adapter": h_adapter,
                       "CoOp+Adapter": h_joint})
```

Il `train_loader=support` non è un dettaglio: senza, il Linear Probe viene
addestrato su 21.600 immagini e CoOp su 160, e i due finiscono sulla stessa
barra come se fossero confrontabili.

Altri parametri di `run_all_evaluations`:

| Parametro | Cosa fa |
|---|---|
| `extra_models` | Dizionario `{nome: modello_già_addestrato}` |
| `clip_wrapper` | Riusa un backbone già in memoria invece di caricarne un secondo |
| `include_baselines=False` | Valuta **solo** i modelli passati, saltando Zero-Shot / Ensemble / Linear Probe / OT |
| `train_loader` | Dati per il Linear Probe |

### 4.4 Memoria GPU per epoca

`train()` restituisce due serie distinte, che rispondono a domande diverse:

```python
history["gpu_epoch_mb"]   # picco DENTRO l'epoca — confronto equo fra metodi
history["gpu_mb"]         # picco cumulativo dall'inizio — "entra in 5 GB?"
history["peak_gpu_mb"]    # scalare: il massimo dell'intera run
```

```python
plot_memory_vs_epochs(histories, per_epoch=True)    # → plots/memory_vs_epochs.png
plot_memory_vs_epochs(histories, per_epoch=False)   # curva cumulativa, a gradini
```

L'etichetta dell'asse dice quale delle due sta disegnando. Scambiarle produce un
grafico che sembra una perdita di memoria quando non c'è nessuna perdita.

### 4.5 Il modello congiunto

```python
from src.coop_adapter import CoOpAdapterModel

m = CoOpAdapterModel(device="cuda", n_ctx=16, reduction_ratio=4)
m.parameter_breakdown()
# {'context': 8192, 'adapter': 131712, 'total': 139904}
```

Costruttore: `class_names`, `n_ctx`, `class_specific`, `ctx_init` dal lato CoOp;
`reduction_ratio`, `alpha`, `learnable_alpha`, `constrain_alpha` dal lato
adapter. Si addestra e si valuta con lo stesso `engine.train()` /
`engine.evaluate()` di tutti gli altri.

### 4.6 Solo le baseline e i grafici

```bash
python src/engine.py
```

Valuta Zero-Shot, Zero-Shot Ensemble, Linear Probe e Optimal Transport
sull'intero test set, poi scrive `accuracy_vs_params.png`,
`resource_usage.png`, `combined_summary.png`, `confusion_matrices.png`,
`per_class_accuracy.png` in `plots/`.

⚠️ **Sovrascrive i grafici comparativi esistenti**, e non contiene nessuno dei
tre metodi del team — per quello serve il codice della sezione 4.3. Tempo:
~15–20 minuti, di cui la maggior parte è il fit del Linear Probe su tutto
EuroSAT.

---

## 5. Dove finisce cosa

```
plots/                                  ← scritta SOLO da run_all.py
├── unified_results.json                ← la griglia completa + il blocco
│                                          "_config" con protocollo e ricette
├── predictions_<dataset>.npz           ← predizioni e label grezze, così le
│                                          figure si rifanno senza GPU
├── unified_comparison.png              ← accuratezza, con barre d'errore
├── accuracy_vs_params_unified.png      ← LA figura del progetto
├── training_cost.png                   ← il costo che davvero discrimina
├── memory_vs_epochs_<dataset>.png      ← deliverable del brief
├── tsne_features.png                   ← 3 pannelli + silhouette score
├── confusion_matrices.png              ← solo dataset con ≤20 classi
├── per_class_accuracy.png
│
├── coop_results.json                   ← ablation di Carlo, per tag (13 run)
├── coop_sweeps.png
├── coop_training_curves.png
├── clip_adapter_results.json           ← sweep reduction ratio di Marco
├── clip_adapter_alpha_results.json     ← sweep alpha
├── clip_adapter_*_sweep.png
│
└── baselines_only/                     ← `python src/engine.py`, 4 baseline
    ├── accuracy_vs_params.png             su EuroSAT. Tenute separate: sono
    ├── resource_usage.png                 un sanity check, non i risultati
    ├── combined_summary.png               del progetto
    ├── confusion_matrices.png
    └── per_class_accuracy.png

data/eurosat/                           ← 94 MB, in .gitignore
```

> **Perché `baselines_only/`.** `engine.py` e `run_all.py` scrivevano gli
> stessi nomi di file nella stessa cartella: `accuracy_vs_params.png` con
> quattro barre su EuroSAT accanto a `accuracy_vs_params_unified.png` con
> dieci metodi su tre dataset, entrambe aggiornate, nessuna delle due
> etichettata. Ora `plots/` ha un solo autore.

Tutti i percorsi sono assoluti, derivati da `__file__`: qualunque sia la
directory da cui lanci uno script, i file finiscono in `<repo>/plots/`.

> ⚠️ **`src/plots/`**: era una cartella duplicata, residuo di quando i percorsi
> erano relativi. Marco l'ha eliminata lui stesso nel commit `5810187`, spostando
> i suoi quattro file in `plots/`. Ma **nello stesso commit** ha cambiato
> `src/coop_experiments.py` puntando i risultati di CoOp *dentro* la cartella
> appena svuotata. Vedi la sezione 6.4: è un problema da risolvere prima di
> allinearsi a `origin`.

---

## 6. Git

### 6.1 Ispezione — sempre sicuri

```bash
git status --short                      # cosa è cambiato
git diff                                # COSA è cambiato davvero ← leggi questo
git diff --stat                         # riassunto per file
git log --oneline --graph --all -20     # dove siamo rispetto agli altri branch
git show <commit>:<file> | head -40     # com'era un file in un commit

# Dove puntano davvero tutti i branch, locali e remoti, in una riga sola.
# Utile perché i ref remoti cambiano sotto i piedi quando i compagni pushano.
git for-each-ref --format='%(refname:short) %(objectname:short)' refs/heads refs/remotes

# "Il lavoro di X è già dentro il mio HEAD?"  Exit code 0 = sì.
git merge-base --is-ancestor origin/marco HEAD; echo $?
```

**Leggi sempre `git diff`, non `git status`.** Una `M` accanto a un file di
Marco o di Mattia che non hai mai aperto non è un dettaglio: in questa
sessione, dietro due `M` di aspetto innocuo c'erano 290 righe cancellate, cioè
Tip-Adapter, Tip-Adapter-F e LoRA che sparivano.

Per controllare *quale versione* di un file hai su disco — cioè per rispondere
alla domanda "questo file è quello di Marco, o una copia vecchia?":

```bash
echo "disco:  $(git hash-object src/clip_adapter.py)"
echo "HEAD:   $(git show HEAD:src/clip_adapter.py | git hash-object --stdin)"
echo "marco:  $(git show origin/marco:src/clip_adapter.py | git hash-object --stdin)"
```

Hash uguali = contenuto identico. Un hash diverso da `HEAD` significa solo che
hai modifiche non committate; un hash **uguale a un commit vecchio** mentre
`HEAD` ne ha uno più recente è il campanello d'allarme: il file su disco è
tornato indietro.

### 6.2 Ripristino locale — sicuro, non tocca l'indice

```bash
git checkout -- src/clip_adapter.py src/clip_adapter_experiments.py
git checkout -- plots/                  # ridà i grafici committati
```

Usalo quando un file altrui è tornato indietro, o quando una run di prova ha
sovrascritto un grafico buono.

### 6.3 Il comando in sospeso — MODIFICA L'INDICE

Non è stato eseguito. Va lanciato a mano, una volta sola:

```bash
# Normalizza i line ending, dopo aver aggiunto .gitattributes.
# Elimina il rumore per cui git segnala come modificati file identici.
git add --renormalize .
```

Poi serve un commit, che decidi tu.

Il secondo comando che era in lista, `git rm -r --cached src/plots`, **non
serve più**: Marco ha eliminato lui la cartella duplicata nel commit `5810187`.

### 6.4 ⚠️ Situazione al 2026-09-01, ore 10:25 — Marco ha pushato

Durante questa sessione i ref remoti sono cambiati. **Tutti** i branch remoti
puntano ora allo stesso commit:

```
carlo          8706c3f        ← locale, indietro
main           4129212        ← locale, indietro di 1 commit
origin/carlo   5810187
origin/main    5810187
origin/marco   5810187
origin/mattia  5810187
```

`5810187` — *"Small fixes to filetree"*, di Marco — fa tre cose:

1. **sposta `src/plots/*` in `plots/*`** — la cartella duplicata sparisce, ed
   è giusto;
2. rigenera i cinque grafici comparativi;
3. cambia **una riga in `src/coop_experiments.py`**, che è un file di Carlo.

**È la terza a essere un problema.** La riga cambiata è:

```diff
-RESULTS_DIR = PROJECT_ROOT / "plots"
+RESULTS_DIR = PROJECT_ROOT / "src/plots"
```

Nello stesso commit in cui `src/plots/` viene svuotata, i risultati di CoOp
vengono puntati lì dentro. Ma `plots/coop_results.json` — le 13 run delle
ablation — resta dov'è, in `plots/`.

Conseguenza concreta, verificata leggendo `load_results()`:

```python
def load_results(path=RESULTS_JSON):
    if not path.exists():
        return []        # ← nessun errore, nessun warning
```

Quindi su `5810187`:

- `python src/coop_experiments.py --plot` **non trova nulla** e ridisegna i
  grafici a partire da zero run, cancellando di fatto le figure buone;
- un nuovo sweep **ricrea `src/plots/`**, cioè esattamente la cartella
  duplicata che lo stesso commit aveva appena eliminato;
- le 13 run esistenti restano sul disco ma escono dai grafici, **senza nessun
  messaggio d'errore**.

**Cosa fare prima di allinearsi a `origin`.** La riga va rimessa a
`PROJECT_ROOT / "plots"`, coerente con lo spostamento che Marco stesso ha
fatto. Da chiedere a lui: probabilmente ha invertito i due lati della modifica.

```bash
# 1. Guardare cosa arriva, senza toccare niente
git fetch
git log --oneline HEAD..origin/main
git diff HEAD origin/main --stat

# 2. Il file di Carlo, riga per riga
git diff HEAD origin/main -- src/coop_experiments.py
```

Il merge vero va fatto **dopo** aver deciso cosa fare delle modifiche non
committate di questa sessione: `origin/main` tocca `src/coop_experiments.py` e
i `plots/*.png`, ed entrambi sono file su cui c'è lavoro locale.

---

## 7. Diagnostica — sintomi e cosa lanciare

| Sintomo | Causa probabile | Cosa fare |
|---|---|---|
| `python src/clip_adapter.py` stampa **solo** il blocco `CLIPAdapterModel` | Il working tree è tornato a una versione precedente al lavoro di Marco | `git diff src/clip_adapter.py` per confermare, poi `git checkout -- src/clip_adapter.py src/clip_adapter_experiments.py` |
| `ctx grad norm: 0` oppure `'NoneType' object has no attribute 'norm'` | Un `@torch.no_grad()` di troppo su un metodo override | Toglierlo. `get_text_features` di CoOp e `get_image_features` dell'adapter **non devono averlo** |
| `ValueError: No trainable parameters found in the model` | Stai passando `TipAdapterModel` a `engine.train()` | È corretto: Tip-Adapter è training-free. Usa `build_cache()`, e `finetune_cache()` per Tip-Adapter-F |
| `AttributeError: object has no attribute 'prompt_learner'` | Hai scritto il modello congiunto per ereditarietà multipla | Usa `CoOpAdapterModel` da `src/coop_adapter.py` |
| L'accuratezza su DTD o Flowers102 è molto più bassa del previsto | Template del prompt sbagliato | `model.build_prompts()[0]` — deve corrispondere al dominio, non essere `"a satellite image of ..."` |
| `git status` segnala un file come modificato ma `git diff` è vuoto | Solo line ending | Innocuo. Si sistema con `git add --renormalize .` |
| `CUDA out of memory` | Due backbone caricati insieme | Passa `clip_wrapper=` a `run_all_evaluations`, e `del model; torch.cuda.empty_cache()` fra un modello e l'altro |
| Un grafico comparativo ha numeri strani e bassi | L'ha sovrascritto una run di prova su un sottoinsieme | `git checkout -- plots/` |
| `coop_sweeps.png` esce vuoto o con pochissimi punti | `RESULTS_DIR` punta a una cartella dove `coop_results.json` non c'è; `load_results()` restituisce `[]` in silenzio | `python -c "from src.coop_experiments import RESULTS_JSON; print(RESULTS_JSON, RESULTS_JSON.exists())"` — deve essere `<repo>/plots/coop_results.json True`. Vedi sezione 6.4 |

### Verificare l'ambiente

```bash
python -c "import torch, open_clip, ot, sklearn; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0)); print('open_clip', open_clip.__version__, '| POT', ot.__version__)"
```

Atteso su questa macchina:

```
torch 2.5.1+cu121 | cuda True | Quadro P2200
open_clip 3.3.0 | POT 0.9.7.post1
```

⚠️ **Non aggiornare torch.** La GPU è Pascal (sm_61) e le build recenti di
PyTorch non compilano più per quell'architettura: il pin su cu121 è deliberato.

### Compilare tutto senza eseguire

```bash
python -m py_compile src/*.py && echo "COMPILE OK"
```

Controllo di sintassi su tutti i moduli in un paio di secondi. Utile subito
dopo un merge, prima di lanciare qualsiasi cosa che costi tempo.
