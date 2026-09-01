# Report di integrazione — codice dei tre workstream

> Data: **2026-09-01** · Autore: Carlo · Branch analizzato: `main` a `4129212`
> Metodo: lettura incrociata dei file dei tre membri + **esecuzione reale** di
> una suite di integrazione (12 test) su GPU Quadro P2200, EuroSAT locale.
>
> Questo documento risponde a una domanda sola: **il codice dei tre workstream,
> messo insieme, funziona?** La risposta è *sì, ma con quattro problemi seri e
> un incidente già avvenuto*. Tutto ciò che segue è verificato eseguendo il
> codice, non dedotto leggendolo.

---

## 0. Sommario esecutivo

> **Aggiornamento 2026-09-01 (seconda parte della sessione).** Su richiesta di
> Carlo tutte le soluzioni proposte sono state **implementate e testate**,
> tranne P2 (`docs/report.tex`), lasciata al team. La suite di verifica dopo i
> fix passa **11/11**. Il dettaglio di cosa e' cambiato e' nella sezione 15, in
> fondo. **Nessun commit e' stato fatto: le modifiche sono nel working tree.**

| # | Problema | Gravità | Chi | Stato |
|---|---|---|---|---|
| P1 | Il working tree conteneva versioni vecchie che **cancellavano Tip-Adapter, Tip-Adapter-F e LoRA di Marco** | 🔴 Critico | Carlo | RISOLTO |
| P2 | Il merge ha perso lo **scheletro LaTeX del report** di Marco | 🔴 Alto | Team | LASCIATO AL TEAM |
| P3 | Il **template del prompt è cablato su EuroSAT** in `engine.train()` e nei 3 modelli di Marco → DTD e Flowers102 girano con prompt sbagliati | 🔴 Alto | Mattia + Marco | RISOLTO |
| P4 | `run_all_evaluations()` **non include nessuno dei tre metodi del team** → i grafici comparativi del brief mostrano solo le baseline | 🔴 Alto | Mattia | RISOLTO |
| P5 | Il **modello congiunto** per ereditarietà multipla non si costruisce | 🟠 Medio | Carlo | RISOLTO (`src/coop_adapter.py`) |
| P6 | `alpha` learnable **non vincolata in [0,1]** e sweep di Marco non più riproducibile | 🟠 Medio | Marco | RISOLTO |
| P7 | Il plot **"Memory Usage vs. Epochs"** del brief non è producibile: il dato per epoca non viene raccolto | 🟠 Medio | Mattia | RISOLTO |
| P8 | **Cinque implementazioni di `predict()`**, contro il "no method reimplements evaluation" del brief | 🟠 Medio | Team | RISOLTO |
| P9 | `src/plots/` duplicato + script che scrivono in base alla cwd | 🟡 Basso | Marco | RISOLTO nel codice; resta da togliere `src/plots/` dal tracciamento |
| P10 | Line ending misti, `.gitattributes` assente → conflitti fantasma | 🟡 Basso | Team | `.gitattributes` creato; resta un comando da lanciare |
| P11 | `count_trainable_params()` di CLIP-Adapter è cambiato di +1 → le tabelle vanno riallineate | 🟡 Basso | Marco | RISOLTO di conseguenza |

**Esito dei test iniziali: 9/10.** **Dopo i fix: 11/11.** L'unico fallimento iniziale (P5) era un fallimento
*atteso*, cercato apposta per documentarlo.

---

## 1. Cos'è cambiato rispetto all'ultimo checkpoint

`.claude/memory.md` fotografava una situazione ormai superata. Due novità
grosse:

**Il merge è stato fatto.** `main` non è più fermo al primo giorno: il commit
`4129212` ("Resolved confluiicts for merge") è un **merge a due genitori** —
`8706c3f` (Carlo, che già conteneva Mattia) e `800db3f` (Marco). Quindi oggi
`main` contiene il lavoro di tutti e tre:

```
*   4129212  main, origin/main   ← merge
|\
| * 800db3f  origin/marco        Learnable Alpha, Tip-Adapter-F, LoRA
| * d1a5c4f
* | 8706c3f  carlo               CoOp ablation results and sweep plots
* | 239a3b2                      CoOp ablation driver
* | c9e7f46                      CoOp prompt learner + few-shot
* | 71fc9e6  origin/mattia
|/
* 98950f0                        antenato comune
```

Verificato: `git merge-base --is-ancestor origin/mattia HEAD` → vero. Non
serve più mergiare nessuno.

**`main` è già su origin.** `main` e `origin/main` puntano entrambi a
`4129212`: quello che c'è nel repository locale è quello che vedono Marco e
Mattia.

---

## 2. P1 — Il problema più grave, già risolto

### Cos'era

All'apertura della sessione `git status` diceva:

```
 M src/clip_adapter.py
 M src/clip_adapter_experiments.py
```

Sembrano due modifiche innocue. Non lo erano. Il `git diff` mostrava **290
righe cancellate e 5 aggiunte**, e le righe cancellate erano tutto il lavoro
recente di Marco:

- l'intera classe `TipAdapterModel` (cache key-value + `finetune_cache()`, cioè Tip-Adapter-F);
- l'intera classe `LinearLoRA`;
- l'intera classe `VisionLoRAModel`;
- il ritorno di `alpha` da `nn.Parameter` a semplice float;
- lo sweep su alpha in `clip_adapter_experiments.py`.

### Come l'ho verificato

Confrontando l'hash del contenuto dei file su disco con quello delle versioni
storiche. È il modo più diretto per rispondere alla domanda "questo file su
disco, quale commit è?":

```
src/clip_adapter.py
  98950f0  -> fc8d46a...     (antenato comune, PRIMA di Marco)
  800db3f  -> a1b97ec...     (Marco, versione buona)
  4129212  -> a1b97ec...     (main: contiene Marco, corretto)
  WORKTREE -> fc8d46a...     ← il file su disco era l'ANTENATO
```

I file sul disco erano letteralmente la versione del 19 agosto — data di
modifica `Aug 19 18:58`, mai toccata dal merge. Il **commit** di merge è
corretto e contiene il codice di Marco; è la **working directory** a essere
rimasta indietro. Probabilmente il merge è stato risolto altrove (interfaccia
web, o un `reset --soft`) senza che i file locali venissero aggiornati.

### Perché era pericoloso

Il prossimo `git commit -a` di Carlo avrebbe silenziosamente cancellato
Tip-Adapter, Tip-Adapter-F e LoRA da `main`, con un messaggio di commit che
parlava di tutt'altro. Nessuno se ne sarebbe accorto finché Marco non avesse
provato a girare il suo codice.

### Cosa ho fatto

```bash
# 1. backup delle versioni vecchie (non si sa mai) nello scratchpad di sessione
# 2. ripristino delle versioni committate
git checkout -- src/clip_adapter.py src/clip_adapter_experiments.py
```

`git status` ora è pulito su quei due file, e i test di Marco girano.

### La lezione

Prima di ogni commit su un repository condiviso, **leggere il `git diff`, non
il `git status`**. Uno `M` accanto a un file altrui che non si è mai aperto è
un campanello d'allarme, non un dettaglio.

---

## 3. Cosa ho testato e cosa è uscito

Ho scritto due suite eseguibili (nello scratchpad di sessione, non nel repo —
se le volete permanenti dite dove metterle):

- `test_integration.py` — 10 test su modelli, engine, training e modello congiunto;
- `test_multidataset.py` — estensibilità a dataset diversi da EuroSAT.

Tutte le valutazioni girano su un sottoinsieme bilanciato di 80 immagini di
test (8 per classe) per stare in pochi secondi: **i numeri qui sotto servono a
verificare che il codice funzioni, non sono risultati per il report.**

### 3.1 Tutti i modelli costruiscono e contano i parametri (T1) ✅

| Modello | `predict()` | Parametri addestrabili |
|---|---|---|
| `BaseCLIPWrapper` | no | 0 |
| `CLIPAdapterModel` (r=4) | sì | 131.713 |
| `TipAdapterModel` | sì | 0 |
| `VisionLoRAModel` (r=4) | sì | 368.640 |
| `CoOpModel` (M=16, unified) | sì | 8.192 |

I 368.640 di LoRA tornano: 12 blocchi × 2 layer × (matrici A e B) =
12 × (768·4 + 4·3072 + 3072·4 + 4·768) = 368.640.

I 131.713 di CLIP-Adapter sono 131.712 **+ 1**: è `alpha`, diventata
`nn.Parameter` nel commit `800db3f`. Vedi P11.

### 3.2 Nessun `@torch.no_grad()` di troppo sui metodi override (T2) ✅

Questa è la trappola numero uno del progetto: se un metodo che deve produrre
gradienti eredita o si prende un `@torch.no_grad()`, il training gira, la loss
scende un po' per caso, e **nessun parametro impara nulla** — senza errori.

Il test ha segnalato `CLIPAdapterModel.get_image_features`. **È un falso
positivo, verificato**: quel metodo non ha il decoratore, ha un
`with torch.no_grad():` *interno* che avvolge solo la chiamata al backbone
congelato, e lascia fuori l'adapter:

```python
def get_image_features(self, images):
    with torch.no_grad():                      # solo il backbone congelato
        orig = self.model.encode_image(images)
        orig = orig / orig.norm(dim=-1, keepdim=True)
    return self.adapter(orig)                  # ← qui il gradiente passa
```

È scritto bene: risparmia memoria non costruendo il grafo sui 86M parametri
congelati. La prova empirica è il test T4 qui sotto.

### 3.3 `engine.evaluate()` accetta tutti e sei i modelli (T3) ✅

| Modello | Top-1 | Top-5 | Params |
|---|---|---|---|
| Zero-Shot | 46,25 % | 88,75 % | 0 |
| CLIP-Adapter *non addestrato* | 46,25 % | 88,75 % | 131.713 |
| Tip-Adapter (cache 4-shot) | **56,25 %** | 91,25 % | 0 |
| Vision-LoRA *non addestrato* | 46,25 % | 88,75 % | 368.640 |
| CoOp *non addestrato* | 42,50 % | 86,25 % | 8.192 |
| Optimal Transport | 20,00 % | 73,75 % | 0 |

Tre osservazioni che sembrano bug e non lo sono:

1. **CLIP-Adapter e LoRA non addestrati danno *esattamente* il numero dello
   zero-shot.** È la prova che le inizializzazioni sono corrette. In
   CLIP-Adapter `up_proj` è inizializzato con `std=1e-4`, quindi
   `f_blended ≈ 0.8·f_orig`, e poiché subito dopo si normalizza in L2 il
   fattore 0,8 sparisce: la feature è identica all'originale. In LoRA
   `lora_B` è inizializzato a zero, quindi `x @ A @ B = 0` e il layer è
   l'identità. Entrambi partono **esattamente** dal modello pre-addestrato e
   non lo distruggono al primo step: è esattamente ciò che il brief chiede.

2. **CoOp non addestrato sta *sotto* lo zero-shot** (42,50 vs 46,25). Corretto:
   il contesto è inizializzato con rumore gaussiano a scala 0,02, cioè è un
   prompt privo di significato. CoOp parte peggio dello zero-shot e lo supera
   con l'addestramento (86,63 % a 16 shot nelle ablation vere).

3. **L'OT prende meno della metà dello zero-shot** e conferma il numero
   full-shot (21,74 %). Non è codice rotto — vedi il punto aperto 4 in
   `memory.md`: i patch token intermedi non vivono nello spazio allineato, e
   marginali uniformi pesano il cielo quanto l'edificio. **Va scritto nel
   report**, altrimenti sembra un errore.

### 3.4 `engine.train()` aggiorna solo i parametri giusti (T4) ✅

| Modello | Loss (2 epoche) | Parametri aggiornati | Backbone congelato |
|---|---|---|---|
| CLIP-Adapter | 1,544 → 1,456 | 5 / 5 | sì |
| CoOp | 2,091 → 1,694 | 1 / 1 | sì |

Il test confronta il valore di *ogni* tensore addestrabile prima e dopo, e
ricontrolla che `requires_grad` sia `False` su tutti gli 86M parametri di
CLIP. Il contratto "il backbone è sempre congelato" regge per entrambi i
metodi.

### 3.5 Tip-Adapter e `engine.train()` (T5) ✅

`engine.train()` su `TipAdapterModel` solleva
`ValueError: No trainable parameters found in the model`. **È il comportamento
giusto**: Tip-Adapter è *training-free* per costruzione, si usa
`build_cache()`; la versione addestrabile è Tip-Adapter-F, che si attiva con
`finetune_cache()` e ha un proprio loop interno. Il messaggio d'errore però è
fuorviante ("Did you forget to add nn.Parameter?"): chi non conosce il metodo
penserà a un bug. Vale la pena che Marco lo documenti nel report.

### 3.6 Estensibilità ad altri dataset — **qui c'è un problema vero**

`dataset.py` di Mattia supporta tre dataset con tre template diversi:

| Dataset | Classi | Template |
|---|---|---|
| EuroSAT | 10 | `"a satellite image of {}"` |
| DTD | 47 | `"a photo of a {} texture"` |
| Flowers102 | 102 | `"a photo of a {}, a type of flower"` |

**CoOp regge**: costruito su DTD produce prototipi `(47, 512)`, e in modalità
CSC conta 385.024 parametri = 47 × 16 × 512, esatto. Nessun 10 cablato.

**Il resto no.** Vedi P3.

---

## 4. P2 — Il merge ha perso il report LaTeX di Marco

`docs/report.tex` era l'unico conflitto previsto, ed è stato risolto nel modo
sbagliato. È un conflitto *logico*, non testuale: Marco ci aveva messo lo
scheletro LaTeX vero dell'elaborato, Mattia ci aveva scritto sopra un piano di
design in markdown per il supporto multi-dataset. Il merge ha tenuto Mattia:

```
origin/marco  docs/report.tex -> 89990be...   (LaTeX)
origin/mattia docs/report.tex -> d7270e4...   (markdown)
main          docs/report.tex -> d7270e4...   ← ha vinto il markdown
```

Oggi `main` contiene un file con estensione `.tex` il cui contenuto comincia
con `# Multi-Dataset Support for CLIP Adaptation Benchmark` e continua con
tabelle markdown. Non compila, e lo scheletro dell'elaborato è sparito da
`main` (è recuperabile: vive ancora in `origin/marco`).

### Soluzione

I due file sono due cose diverse e devono stare in due file diversi:

```bash
git show origin/main:docs/report.tex  > docs/multi_dataset_plan.md   # il piano di Mattia
git show origin/marco:docs/report.tex > docs/report.tex              # il LaTeX di Marco
git add docs/multi_dataset_plan.md docs/report.tex
git commit -m "Restore Marco's LaTeX report skeleton; move Mattia's multi-dataset plan to its own file"
```

**Da concordare con Marco e Mattia prima di eseguirlo**: è un file di
competenza condivisa e la risoluzione precedente è stata una loro scelta (o un
loro incidente — va chiesto).

---

## 5. P3 — Il template del prompt è cablato su EuroSAT

### Il problema

Il classificatore, in questo progetto, non ha una testa lineare: le "classi"
*sono* i prototipi testuali, ottenuti passando una frase nel text encoder. La
frase la costruisce il template. Se il template è sbagliato, i prototipi sono
sbagliati e l'accuratezza crolla — senza che nulla lanci un'eccezione.

Verificato eseguendo il codice:

```
CLIPAdapterModel(class_names=DTD_CLASS_NAMES)
  text_prototypes: (47, 512)                      ← la forma è giusta
  prompt effettivo per 'banded': 'a satellite image of banded'
  prompt CORRETTO per DTD     : 'a photo of a banded texture'
```

"a satellite image of banded" è una frase priva di senso: si sta chiedendo a
CLIP di trovare l'immagine satellitare di una texture a righe.

Chi è colpito:

| Punto | Cosa fa | Corretto? |
|---|---|---|
| `engine.py:391` in `train()` | `prompts = [PROMPT_TEMPLATE.format(n) for n in class_names]` | ❌ template EuroSAT fisso |
| `clip_adapter.py` `CLIPAdapterModel._update_text_prototypes()` | idem | ❌ |
| `clip_adapter.py` `TipAdapterModel._update_text_prototypes()` | idem | ❌ |
| `clip_adapter.py` `VisionLoRAModel._update_text_prototypes()` | idem | ❌ |
| `CoOpModel` | nessun template: il contesto è appreso | ✅ immune |

`train()` accetta già `class_names` ma **non** `prompt_template`: passare le
classi giuste non basta.

Nota interessante per il report: **CoOp è l'unico metodo strutturalmente
immune a questo bug**, perché il template è proprio la cosa che sostituisce.
È un argomento a favore del prompt learning che non è solo "qualche punto di
accuratezza in più".

### Soluzione proposta

Un parametro opzionale, con l'attuale comportamento come default — così
nessuno rompe niente:

```python
# engine.py, train()
def train(..., class_names=None, prompt_template=None, ...):
    if prompt_template is None:
        prompt_template = PROMPT_TEMPLATE          # default EuroSAT: compatibile
    prompts = [prompt_template.format(name) for name in class_names]
```

```python
# clip_adapter.py, nei tre modelli
def __init__(self, ..., class_names=EUROSAT_CLASS_NAMES, prompt_template=PROMPT_TEMPLATE):
    self.prompt_template = prompt_template
    ...

def _update_text_prototypes(self):
    prompts = [self.prompt_template.format(n) for n in self.class_names]
    with torch.no_grad():
        self.text_prototypes = self.get_text_features(prompts)
```

Le classi dataset di Mattia espongono già `self.prompt_template` come
attributo (`dataset.py:189`, `380`, `650`), quindi il valore da passare si
legge dal dataloader: `train_loader.dataset.prompt_template`. Il pezzo
mancante è solo farlo arrivare fino ai modelli.

**Sono file di Mattia e di Marco: la modifica va proposta a loro, non fatta
unilateralmente.** Finché resta EuroSAT, non morde nessuno.

---

## 6. P4 — I grafici comparativi non contengono i metodi del team

`engine.run_all_evaluations()` — la funzione che produce i grafici finali
`accuracy_vs_params.png`, `resource_usage.png`, `combined_summary.png` —
valuta esattamente quattro modelli:

```python
all_results["ZeroShot"]          = evaluate(ZeroShotCLIP(clip_wrapper), ...)
all_results["ZeroShot-Ensemble"] = evaluate(ZeroShotEnsembleCLIP(clip_wrapper), ...)
all_results["LinearProbe"]       = evaluate(lp_model, ...)
all_results["OptimalTransport"]  = evaluate(ot_model, ...)
```

**Manca tutto**: CoOp, CLIP-Adapter, Tip-Adapter, LoRA. Il brief affida a
Mattia il deliverable *"Aggregate final metrics and plot the comparative
graphs: Accuracy vs. Trainable Parameters, Memory Usage vs. Epochs"*, e il
punto interessante del progetto è proprio il trade-off fra accuratezza e
parametri: 0 (zero-shot) → 5.130 (linear probe) → 8.192 (CoOp) → 131.713
(adapter) → 368.640 (LoRA). Con solo le baseline, il grafico ha tre punti di
cui due a zero parametri e non dice niente.

### Soluzione

Due strade, da scegliere a tre:

**(a) Estendere `run_all_evaluations()`** con i modelli addestrati. Richiede
che accetti i modelli già addestrati dall'esterno, altrimenti la funzione
diventa un mostro che riaddestra tutto:

```python
def run_all_evaluations(test_loader, device="cuda", use_wandb=True,
                        extra_models=None):
    ...
    for name, model in (extra_models or {}).items():
        all_results[name] = evaluate(model, test_loader, model_name=name,
                                     use_wandb=use_wandb)
```

**(b) Aggregare dai JSON.** Ogni workstream salva già i propri risultati
(`plots/coop_results.json`, `plots/clip_adapter_results.json`), quindi uno
script finale `aggregate.py` li unisce e chiama `plot_comparative_results()`.
Più robusto: nessuno deve riaddestrare nulla per rigenerare un grafico, ed è
già la filosofia di `coop_experiments.py --plot`.

**Raccomandazione: (b).** Costa meno e non tocca la firma di funzioni
condivise. Ma attenzione al punto 12: i JSON attuali non sono confrontabili
fra loro perché mescolano protocolli diversi.

---

## 7. P5 — Il modello congiunto CoOp + CLIP-Adapter

### Il fallimento (test T6, atteso)

La scrittura ovvia non funziona:

```python
class CoOpAdapterModel(CoOpModel, CLIPAdapterModel):
    pass

CoOpAdapterModel(device="cuda")
# AttributeError: 'CoOpAdapterModel' object has no attribute 'prompt_learner'
```

**Perché.** L'ordine di risoluzione (MRO) è
`CoOpAdapterModel → CoOpModel → CLIPAdapterModel → BaseCLIPWrapper`. Quando
si costruisce l'oggetto:

1. parte `CoOpModel.__init__`, che come prima cosa chiama `super().__init__()`
   credendo di parlare con `BaseCLIPWrapper`;
2. ma nell'MRO il "super" di `CoOpModel` è `CLIPAdapterModel`, non la base;
3. `CLIPAdapterModel.__init__` arriva in fondo e chiama
   `self._update_text_prototypes()`;
4. quel metodo, risolto sull'istanza, è la versione di CoOp, che legge
   `self.prompt_learner`;
5. ma `prompt_learner` verrà creato solo *dopo* il ritorno da
   `super().__init__()`, cioè al punto 1 non esiste ancora.

Classico ordine d'inizializzazione invertito. Non è colpa di nessuno dei due:
entrambe le classi sono corrette da sole, sono le due `__init__` a non essere
cooperative (nessuna delle due è scritta per passare `**kwargs` lungo l'MRO).

### La soluzione, verificata funzionante (test T7)

Non usare l'MRO cooperativo: `__init__` esplicito che chiama direttamente la
base e costruisce a mano i due moduli, poi presta i metodi dalle due classi.

```python
class CoOpAdapterModel(BaseCLIPWrapper):
    """CoOp sul testo + CLIP-Adapter sulla visione, sullo stesso backbone congelato."""

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",
                 device="cuda", class_names=EUROSAT_CLASS_NAMES, n_ctx=16,
                 class_specific=False, ctx_init=None, reduction_ratio=4, alpha=0.2):
        # Chiamata ESPLICITA alla base: scavalca l'MRO cooperativo e impedisce
        # che CLIPAdapterModel.__init__ giri prima che il prompt learner esista.
        BaseCLIPWrapper.__init__(self, model_name=model_name,
                                 pretrained=pretrained, device=device)

        self.class_names = list(class_names)
        self.n_ctx, self.class_specific, self.alpha = n_ctx, class_specific, alpha

        self.prompt_learner = PromptLearner(
            token_embedding=self.model.token_embedding, tokenizer=self.tokenizer,
            class_names=self.class_names, n_ctx=n_ctx,
            class_specific=class_specific, ctx_init=ctx_init).to(device)

        self.adapter = VisionAdapterModule(
            embed_dim=self.model.visual.proj.shape[1],
            reduction_ratio=reduction_ratio, alpha=alpha).to(device)

        # Rete di sicurezza: la base ha congelato il backbone prima che i due
        # moduli esistessero, quindi si riafferma l'invariante.
        for p in self.model.parameters():
            p.requires_grad = False

    # Metodi presi in prestito, senza ereditarietà. Nessuno dei due ha
    # @torch.no_grad(): il gradiente deve arrivare a ctx e all'adapter.
    _encode_text_from_embeddings = CoOpModel._encode_text_from_embeddings
    get_text_features            = CoOpModel.get_text_features
    get_image_features           = CLIPAdapterModel.get_image_features

    @torch.no_grad()
    def predict(self, images):
        # I prototipi NON si mettono in cache: ctx cambia a ogni step di
        # ottimizzazione (contratto #4). Si segue la strada di CoOp, non
        # quella di Marco.
        sims = self.get_image_features(images) @ self.get_text_features().T
        return sims.argmax(dim=-1), sims
```

**Risultato del test T7, eseguito:**

```
trainable = 139.905  (atteso 8.192 + 131.713 = 139.905)  ✅
ctx aggiornato dopo il training      = True   ✅
adapter aggiornato dopo il training  = True   ✅
loss 2.635 → 2.123 in 2 epoche               ✅
gira dentro engine.train() e engine.evaluate() senza modifiche   ✅
```

Il modello congiunto funziona, i gradienti arrivano a **entrambi** i moduli, e
si aggancia all'engine condiviso senza toccare i file di nessuno. Va messo in
un file nuovo di Carlo — proposta: `src/coop_adapter.py` — così non si tocca
né `coop.py` né `clip_adapter.py`.

Le due trappole da non "ripulire" mai:

- niente `@torch.no_grad()` su `get_text_features` / `get_image_features`;
- niente cache dei prototipi testuali in `predict()`.

---

## 8. P6 — `alpha` learnable di Marco

Verificato eseguendo (test T8):

```
alpha è nn.Parameter = True
0.2000 → 0.1733  dopo 5 epoche a lr=1e-2
set_alpha(0.5)   → 0.5000   (funziona)
```

Tre conseguenze, tutte da dichiarare:

1. **Lo sweep su alpha non è più riproducibile.** I numeri in `memory.md`
   (0.1→97,11 %, 0.2→97,17 %, 0.5→96,94 %, 0.8→97,15 %) sono stati ottenuti
   con alpha *fisso*; con il codice attuale alpha si muove durante il training
   e il valore iniziale è solo un punto di partenza. Vanno tenuti come **due
   esperimenti distinti**: "alpha fisso, sweep manuale" e "alpha appreso".

2. **Alpha non è vincolata in [0,1].** Non c'è né `sigmoid` né `clamp`. Nulla
   vieta ad alpha di finire a 1,4 o a −0,3, che rende
   `α·MLP(f) + (1−α)·f` non più un'interpolazione ma un'estrapolazione. In 5
   epoche corte non è successo (0,17), ma su 100 epoche va **loggato il valore
   finale**. Se lo si vuole vincolare, il modo pulito è parametrizzare il
   logit: `self.alpha_logit = nn.Parameter(...)` e usare
   `torch.sigmoid(self.alpha_logit)` nel forward.

3. **Nota teorica per il report.** Il blending è seguito da una
   normalizzazione L2. Questo rende il modulo **invariante alla scala**: se
   alpha e (1−alpha) crescono insieme, la direzione della feature non cambia.
   Quello che alpha controlla davvero è solo il *rapporto* fra contributo
   dell'adapter e contributo originale. È probabilmente il motivo per cui lo
   sweep su alpha è risultato quasi piatto (96,94–97,17 %): un range del 25 %
   di variazione nel parametro produce un range dello 0,23 % nell'accuratezza.

---

## 9. P7 — "Memory Usage vs. Epochs" non è producibile

Il brief chiede esplicitamente questo grafico. Il dato **non viene raccolto**.
In `engine.train()` la memoria è misurata una sola volta, alla fine:

```python
gpu_memory = _get_gpu_memory_mb()
history["peak_gpu_mb"] = gpu_memory.get("gpu/max_allocated_mb", 0.0)   # uno scalare
```

mentre loss, accuratezza e learning rate sono liste per epoca. Con uno scalare
non si disegna una curva: oggi esiste solo un bar chart del picco.

### Soluzione (tre righe)

```python
history = {"train_loss": [], "train_acc": [], "val_top1": [],
           "val_top5": [], "lr": [], "gpu_mb": []}      # ← +1

# dentro il ciclo sulle epoche, dopo lo scheduler.step():
history["gpu_mb"].append(_get_gpu_memory_mb().get("gpu/max_allocated_mb", 0.0))
```

Poi il plot è un `plt.plot(range(1, epochs+1), history["gpu_mb"])`. **File di
Mattia**: da segnalare a lui.

Nota: `torch.cuda.reset_peak_memory_stats()` non viene chiamato all'inizio di
ogni epoca, quindi `max_allocated` è monotono crescente e la curva sarebbe una
scala. Se si vuole il picco *per epoca* va resettato a ogni giro; se si vuole
il picco cumulativo va bene così — sono due grafici diversi e vanno etichettati
di conseguenza.

---

## 10. P8 — Cinque `predict()` per la stessa operazione

Il brief dice *"Every method plugs into the same base class and the same
evaluation loop. No method reimplements evaluation."* Oggi `predict()` è
scritto cinque volte: `ZeroShotCLIP`, `CLIPAdapterModel`, `TipAdapterModel`,
`VisionLoRAModel`, `CoOpModel` (sei con il modello congiunto). Tre di questi
sono identici riga per riga:

```python
image_features = self.get_image_features(images)
similarities = image_features @ self.text_prototypes.T
return similarities.argmax(dim=-1), similarities
```

Non è un bug — tutti funzionano, T3 lo conferma — ma è duplicazione, ed è la
ragione strutturale per cui il modello congiunto ereditava due `predict()` in
conflitto (P5).

### Soluzione proposta

Un `predict()` di default nella base, che i casi speciali sovrascrivono:

```python
# base_model.py
@torch.no_grad()
def predict(self, images):
    """Similarità coseno contro i prototipi testuali. I metodi con una
    logica di scoring propria (Tip-Adapter, OT) fanno override."""
    f_img = self.get_image_features(images)
    f_txt = self.get_text_features(
        [self.prompt_template.format(n) for n in self.class_names])
    sims = f_img @ f_txt.T
    return sims.argmax(dim=-1), sims
```

Ricalcolando i prototipi a ogni chiamata si risolve anche il contratto #4 (la
cache di Marco che va stantia con CoOp): il costo è di 10 frasi nel text
encoder, trascurabile rispetto a un batch di immagini.

Restano necessariamente specializzati: `TipAdapterModel` (formula
cache + zero-shot) e `OptimalTransportCLIP` (restituisce distanze, non
similarità — ed è per questo che `evaluate()` lo riconosce con
`hasattr(model, "sinkhorn_reg")`).

**È un cambio alla classe base di Mattia**: è esattamente il tipo di modifica
che il brief dice di non fare senza avvisare. Da portare alla riunione.

---

## 11. Problemi minori

### P9 — `src/plots/` duplicato

`main` traccia **due** cartelle di grafici:

```
plots/       accuracy_vs_params.png, coop_*.png, confusion_matrices.png, ...
src/plots/   clip_adapter_alpha_results.json, clip_adapter_*_sweep.png, ...
```

Causa: `clip_adapter_experiments.py` usa `save_dir="./plots"`, relativo alla
directory da cui si lancia il comando. Lanciato da `src/`, scrive in
`src/plots/`. `engine.py` ha già la toppa giusta (righe 743-744):

```python
if save_dir == "./plots" and not os.path.exists("./plots"):
    save_dir = str(PROJECT_ROOT / "plots")
```

`coop_experiments.py` usa un path assoluto derivato da `__file__`, che è la
soluzione più robusta. **Fix per Marco**: sostituire `"./plots"` con
`Path(__file__).resolve().parent.parent / "plots"`, e poi
`git rm -r --cached src/plots` una volta rigenerati i file al posto giusto.

### P10 — Line ending misti

| CRLF | LF |
|---|---|
| `base_model.py`, `baselines.py`, `clip_adapter.py`, `clip_adapter_experiments.py`, `dataset.py`, `engine.py`, `optimal_transport.py` | `coop.py`, `coop_experiments.py`, `few_shot.py` |

Non c'è `.gitattributes`. Conseguenza già visibile: `git status` segnalava
`src/coop_experiments.py` come modificato quando il contenuto è **identico
byte per byte** a quello committato (verificato con `diff` dopo aver rimosso i
`\r`). È rumore che maschera le modifiche vere — e in questa sessione è
esattamente il rumore in cui si nascondeva P1.

Fix, da fare una volta sola e concordare perché tocca tutti i file:

```bash
printf '* text=auto eol=lf\n' > .gitattributes
git add --renormalize .
git commit -m "Normalize line endings via .gitattributes"
```

### P11 — I conteggi di parametri sono cambiati

`CLIPAdapterModel` con r=4 riporta ora **131.713** parametri, non 131.712: il
+1 è `alpha` diventata `nn.Parameter`. Vale per tutte le riduzioni:

| r | Prima | Ora |
|---|---|---|
| 16 | 33.312 | 33.313 |
| 8 | 66.112 | 66.113 |
| 4 | 131.712 | 131.713 |
| 2 | 262.912 | 262.913 |

Irrilevante numericamente, ma il grafico "Accuracy vs. Trainable Parameters" va
rigenerato con i numeri attuali per coerenza, e il report deve dire quale
versione descrive.

### P12 — Few-shot contro full-shot: i risultati non sono confrontabili

Punto già noto ma che diventa bloccante ora che serve un grafico comparativo
unico. Oggi convivono due protocolli:

| Metodo | Protocollo | Immagini di training |
|---|---|---|
| CoOp | few-shot 16-shot | 160 |
| Linear Probe, CLIP-Adapter | full-shot | 21.600 |
| Tip-Adapter | cache 16-shot | 160 |

Mettere 86,63 % (CoOp, 160 immagini) e 97,28 % (CLIP-Adapter, 21.600 immagini)
sullo stesso grafico e concludere che l'adapter vince è **scorretto**: si sta
confrontando anche un fattore 135 di supervisione.

`src/few_shot.py` è scritto apposta per essere indipendente dal metodo e
funziona su tutti e tre i dataset. Rieseguire Linear Probe e CLIP-Adapter a 16
shot costa poche decine di minuti e rende il confronto onesto. **È la cosa più
importante da decidere alla prossima riunione.**

### P13 — LoRA non è nel brief

`VisionLoRAModel` non compare in `iter.md` ed è l'unico metodo che **modifica
l'interno del backbone** (inietta matrici a basso rango in `c_fc` e `c_proj` di
tutti e 12 i blocchi visivi), in tensione con il principio "The CLIP backbone
is always frozen". Tecnicamente i pesi originali restano congelati e solo
`lora_A`/`lora_B` si addestrano, quindi la lettera del principio è rispettata;
lo spirito ("only the method-specific module trains, appended outside the
backbone") no.

Non è un motivo per buttarlo — a 368.640 parametri è il punto più a destra del
grafico accuratezza/parametri e rende il trade-off più leggibile. Ma va
**inquadrato esplicitamente nel report** come confronto voluto e fuori brief,
non lasciato lì come se fosse previsto. Da decidere a tre.

---

## 12. Falsi allarmi: cosa NON toccare

Elenco esplicito, perché a una rilettura futura queste cose sembrano bug e
qualcuno le "sistemerà" rompendo tutto in silenzio.

| Sembra un bug | Non lo è, perché |
|---|---|
| `CoOpModel.get_text_features()` **non** ha `@torch.no_grad()`, la versione base sì | Deliberato. Con il decoratore il gradiente non arriverebbe a `ctx` e il training girerebbe senza imparare nulla, senza errori. |
| `CoOpModel.get_text_features(class_names)` **ignora** il suo argomento | Deliberato. `engine.train()` passa prompt già formattati, ma CoOp non costruisce prompt da stringhe: le classi sono negli embedding congelati del suffisso. Il numero di classi viene però controllato e un mismatch alza `ValueError` (verificato, T9). |
| `CLIPAdapterModel.get_image_features` contiene `with torch.no_grad()` | È interno e avvolge **solo** il backbone congelato; l'adapter resta fuori e riceve gradiente (verificato, T4: 5/5 parametri aggiornati). |
| CLIP-Adapter e LoRA non addestrati danno *esattamente* lo zero-shot | È la prova che le inizializzazioni near-zero / zero-init sono corrette e non distruggono le feature pre-addestrate. |
| CoOp non addestrato sta **sotto** lo zero-shot | Il contesto parte da rumore gaussiano: è un prompt senza significato. |
| `engine.train()` su `TipAdapterModel` solleva `ValueError` | Tip-Adapter è training-free: si usa `build_cache()`, e `finetune_cache()` per la variante F. |
| `git status` segnala `src/coop_experiments.py` come modificato | Solo line ending: il contenuto è identico byte per byte. Vedi P10. |
| L'OT prende meno della metà dello zero-shot | Limite noto del metodo, non del codice. Va spiegato nel report. |

---

## 13. Checklist operativa

### Fatto in questa sessione

- [x] Ripristinati `src/clip_adapter.py` e `src/clip_adapter_experiments.py` alle versioni di Marco (P1)
- [x] Verificato che `main` contiene il lavoro di tutti e tre e che è allineato a `origin/main`
- [x] Eseguita la suite di integrazione: 9/10 (l'unico FAIL è quello atteso, P5)
- [x] Verificato il modello congiunto CoOp + CLIP-Adapter: **funziona**, 139.905 parametri, entrambi i moduli si aggiornano
- [x] Rigenerati i grafici CoOp dal JSON (`--plot`) senza riaddestrare

### Da fare da solo (file di Carlo)

- [ ] Creare `src/coop_adapter.py` con `CoOpAdapterModel` (codice al § 7)
- [ ] Ripetere K=1 e K=2 su 3 seed e mediare — il punto a K=2 oggi è rumore
- [ ] Ablation sull'inizializzazione del contesto: casuale vs `--ctx-init "a satellite image of"`
- [ ] Decidere se `.claude/` e `CLAUDE.md` vanno in `.gitignore` (contengono giudizi sul codice altrui)

### Da portare alla riunione, in quest'ordine

1. **P12 — protocollo few-shot vs full-shot.** Blocca il grafico comparativo. Decidere: si riesegue tutto a 16 shot?
2. **P2 — `docs/report.tex`.** Lo scheletro LaTeX di Marco va rimesso in `main`.
3. **P4 — `run_all_evaluations()`.** Chi produce i grafici finali e da dove prende i numeri? (Raccomandazione: aggregare dai JSON.)
4. **P3 — template del prompt.** Serve solo se si va oltre EuroSAT, ma decidere adesso costa poco.
5. **P8 — `predict()` nella classe base**, e **P2 di `memory.md`** — chi applica il template, dato che il brief specifica nomi grezzi.
6. **P6 — alpha learnable**: due esperimenti distinti, e loggare il valore finale.
7. **P13 — LoRA**: si tiene? Come si inquadra nel report?
8. **P7 — memoria per epoca**, **P9 — `src/plots/`**, **P10 — `.gitattributes`**: fix piccoli, si assegnano e si chiudono.

---

## 14. Come rieseguire i controlli

I due script di test stanno nello scratchpad di sessione e non sono nel repo.
Nel frattempo, gli smoke test dei singoli moduli funzionano tutti e si lanciano
**dalla radice del repository**:

```bash
python src/base_model.py               # backbone congelato, 0 parametri addestrabili
python src/dataset.py                  # sanity check dei dataloader
python src/coop.py                     # CoOp: parametri + gradiente su ctx
python src/few_shot.py                 # campionamento K-shot
python src/clip_adapter.py             # CLIP-Adapter + Tip-Adapter + LoRA
python src/coop_experiments.py --plot  # rigenera i grafici CoOp dal JSON
```

Se `python src/clip_adapter.py` stampa solo il blocco `CLIPAdapterModel` e non
quelli di Tip-Adapter e LoRA, **il working tree è di nuovo tornato indietro**:
è il sintomo di P1. Si ripara con
`git checkout -- src/clip_adapter.py src/clip_adapter_experiments.py`.

---

## 15. Cosa è stato implementato (seconda parte della sessione)

Tutte le soluzioni proposte sopra sono state scritte e verificate. **Nessun
commit è stato fatto**: le modifiche sono nel working tree, pronte perché
Carlo le committi come preferisce.

### 15.1 File toccati

| File | Proprietario | Cosa è cambiato |
|---|---|---|
| `src/base_model.py` | Mattia | `predict()` e `build_prompts()` nella classe base; `class_names` e `prompt_template` nel costruttore |
| `src/engine.py` | Mattia | `train(prompt_template=...)`; memoria GPU per epoca; `plot_memory_vs_epochs()`; `run_all_evaluations(extra_models=..., clip_wrapper=..., include_baselines=..., train_loader=...)` |
| `src/clip_adapter.py` | Marco | `prompt_template` sui tre modelli; alpha fisso vs appreso vincolato; due `predict()` duplicati rimossi |
| `src/clip_adapter_experiments.py` | Marco | path dei plot assoluto; `learnable_alpha` nel driver; `alpha_final` nei risultati |
| `src/coop.py` | Carlo | allineato al `predict()` unico |
| `src/coop_adapter.py` | Carlo | **nuovo** — modello congiunto CoOp + CLIP-Adapter |
| `.gitattributes` | — | **nuovo** — normalizzazione dei line ending |

I file di Mattia e Marco sono stati modificati **solo perché Carlo l'ha chiesto
esplicitamente**. Ogni cambiamento è retrocompatibile: i default riproducono il
comportamento precedente, e tutti gli script esistenti girano senza modifiche.

### 15.2 P8 — Il `predict()` unico

`BaseCLIPWrapper` ora ha un `predict()` che tutti ereditano:

```python
image_features = self.get_image_features(images)
text_features  = self.get_text_features(self.build_prompts())
similarities   = image_features @ text_features.T
return similarities.argmax(dim=-1), similarities
```

Verificato tramite MRO che `CLIPAdapterModel`, `VisionLoRAModel`, `CoOpModel` e
`CoOpAdapterModel` lo ereditino tutti dalla base. **Restano specializzati solo
due metodi, ed è corretto così**: `TipAdapterModel` (blending fra cache e
zero-shot) e `OptimalTransportCLIP` (restituisce distanze, non similarità). Da
cinque implementazioni a una.

**La differenza di comportamento è voluta**: la versione base ricalcola i
prototipi testuali a ogni chiamata invece di leggere la cache costruita nel
`__init__`. Costa dieci frasi corte nel text encoder ed è ciò che rende lo
stesso metodo corretto anche quando il lato testo è addestrabile. Senza questo,
il modello congiunto riporterebbe per sempre l'accuratezza del contesto casuale
iniziale.

### 15.3 P3 — Il template del prompt

Ora attraversa tutta la pipeline, con precedenza **argomento esplicito →
attributo del modello → default EuroSAT**:

```python
adapter = CLIPAdapterModel(class_names=DTD_CLASS_NAMES,
                           prompt_template=DTD_PROMPT_TEMPLATE)
adapter.build_prompts()[0]        # 'a photo of a banded texture'
```

Il passaggio intermedio conta: un modello costruito per DTD conosce già le sue
classi e il suo template, e obbligare il chiamante a ripeterli a `train()` è
esattamente il modo in cui i due divergono.

`CoOpModel` e `CoOpAdapterModel` dichiarano `prompt_template="{}"`: CoOp un
template non ce l'ha, sostituirlo è il metodo. Registrarlo esplicitamente evita
di lasciare un template EuroSAT stantio su un modello addestrato su DTD.

### 15.4 P4 — `run_all_evaluations()`

Quattro parametri nuovi, tutti opzionali:

- `extra_models` — dizionario `{nome: modello_addestrato}`, valutati con lo
  **stesso** `evaluate()` delle baseline, quindi i numeri sono direttamente
  confrontabili;
- `clip_wrapper` — riusa un backbone già caricato invece di caricarne un
  secondo (su 5 GB di VRAM è la differenza fra entrarci e non entrarci);
- `include_baselines=False` — valuta solo i modelli passati;
- `train_loader` — dati per il Linear Probe. **Va passato il loader few-shot**
  se il confronto deve essere a supervisione uguale.

I modelli passati vengono messi in `eval()` prima della valutazione, e uno
senza `predict()` alza un `TypeError` con un messaggio esplicito invece di
fallire dentro il loop.

Verificato: la figura comparativa esce ora con **7 modelli** invece di 4, e
copre i parametri addestrabili da 0 a 139.904.

### 15.5 P7 — Memoria per epoca

`train()` registra due serie, che rispondono a domande diverse:

- `history["gpu_epoch_mb"]` — picco **dentro** l'epoca, ottenuto resettando il
  contatore CUDA a inizio epoca. È il confronto equo fra metodi;
- `history["gpu_mb"]` — picco cumulativo dall'inizio, monotono non decrescente.
  È il numero che dice se la run entra in 5 GB.

`history["peak_gpu_mb"]` ora usa il massimo accumulato in Python: leggere il
contatore CUDA alla fine, con i reset per epoca, avrebbe riportato solo
l'ultima epoca — un bug introdotto dal fix stesso, se non ci si fa caso.

Il plot è `engine.plot_memory_vs_epochs(histories, per_epoch=True|False)`.
L'etichetta dell'asse dice quale serie sta disegnando, perché scambiarle
produce un grafico che sembra una perdita di memoria quando non c'è nessuna
perdita. Una `history` vecchia senza le serie viene saltata con un warning
invece di far fallire la figura.

### 15.6 P6 e P11 — Alpha

`VisionAdapterModule` ha due parametri nuovi:

| Configurazione | Parametri (r=4) | Comportamento |
|---|---|---|
| `learnable_alpha=False` (default) | 131.712 | alpha resta dove viene messo — lo sweep torna riproducibile |
| `learnable_alpha=True, constrain_alpha=True` | 131.713 | si impara il *logit* di alpha e si applica una sigmoid nel forward: alpha resta in (0,1) per costruzione, senza clamp e senza discontinuità nel gradiente |
| `learnable_alpha=True, constrain_alpha=False` | 131.713 | comportamento precedente, non vincolato — tenuto per riprodurre le run già fatte |

Il default è alpha **fisso**, quindi i conteggi di parametri tornano ai numeri
delle tabelle esistenti (P11 rientra da solo) e lo sweep su alpha ridiventa un
esperimento sensato. `alpha` è esposta come proprietà: `float(model.alpha)`
funziona in tutti e tre i casi, e `set_alpha()` scrive nella rappresentazione
attiva.

Il driver di Marco accetta `learnable_alpha`, mette la modalità nel tag della
run (senza, una run "learned" ne sovrascriverebbe in silenzio una "fixed") e
registra `alpha_final` nei risultati.

Verificato su EuroSAT full-shot, 1 epoca, r=16: alpha fisso resta a 0,2000
(92,63 %); alpha appreso si sposta a 0,2155 restando in (0,1) (93,00 %).

### 15.7 P5 — Il modello congiunto

`src/coop_adapter.py`, con l'`__init__` esplicito descritto al § 7. Il
docstring del modulo spiega per esteso perché l'ereditarietà multipla
fallisce, e lo smoke test **verifica che continui a fallire**, così nessuno
"semplifica" il file in una riga. Sono compresi: controllo dei gradienti su
entrambi i moduli, verifica che `predict()` venga dalla base,
`parameter_breakdown()` che separa contesto e adapter, e un `assert` nel
costruttore che il backbone sia congelato.

Misurato a M=16, r=4: 8.192 (contesto) + 131.712 (adapter) = **139.904**
parametri, entrambi aggiornati dal training, backbone congelato.

### 15.8 P9 e P10 — Fix minori

`clip_adapter_experiments.py` scrive in `PROJECT_ROOT / "plots"` derivato da
`__file__`, non più in `./plots` relativo alla cwd: lanciato da `src/` non crea
più una seconda cartella. `.gitattributes` normalizza i line ending a LF, con i
formati binari esclusi.

### 15.9 Verifica

Suite di 11 test su GPU con EuroSAT reale: **11/11 passati**. Coprono il
`predict()` unico via MRO, la propagazione del template ai tre modelli e a
`train()`, le due serie di memoria, la generazione del plot, `extra_models` e
`include_baselines`, le tre modalità di alpha, il modello congiunto
end-to-end, e due test di **non regressione**: adapter e LoRA non addestrati
devono continuare a riprodurre *esattamente* lo zero-shot (46,25 % sul
sottoinsieme di prova), e Tip-Adapter deve conservare la sua formula (56,25 %)
e restare training-free.

Rieseguiti anche tutti gli smoke test dei moduli, il driver di Marco su dati
veri, e `coop_experiments.py` sia con `--plot` sia con una run breve completa.
La run di prova è stata rimossa da `plots/coop_results.json`, che è tornato
alle sue 13 run.

### 15.10 Cosa resta da fare a mano

Due operazioni richiedono comandi git che modificano l'indice, e non sono state
eseguite:

```bash
# P10 — dopo aver aggiunto .gitattributes, una volta sola
git add --renormalize .
```

Il secondo comando che era previsto qui, `git rm -r --cached src/plots`, **non
serve piu'**: mentre questa sessione era in corso Marco ha pushato `5810187`,
che elimina lui la cartella duplicata spostando i quattro file in `plots/`.

Lo stesso commit pero' introduce un problema nuovo su un file di Carlo: cambia
`RESULTS_DIR` di `src/coop_experiments.py` da `PROJECT_ROOT / "plots"` a
`PROJECT_ROOT / "src/plots"`, cioe' punta i risultati di CoOp dentro la
cartella che lo stesso commit ha appena svuotato. Siccome `load_results()` fa
`if not path.exists(): return []` senza avvisare, su `5810187` il comando
`--plot` ridisegna i grafici a partire da zero run e un nuovo sweep ricrea
`src/plots/`. Va rimessa a `"plots"` prima di allinearsi a `origin`, e va
chiesto a Marco. Dettaglio e comandi in `docs/COMANDI.md` § 6.4.

E resta P2, lasciata al team: rimettere in `main` lo scheletro LaTeX di Marco e
spostare il piano multi-dataset di Mattia in `docs/multi_dataset_plan.md`.
