# Overlay Problem: separating overlapping red and green points

**[English](#english) · [Русский](#русский)** · [History of the experiments](#history) · [История экспериментов](#история)

**Interactive site / Интерактивный сайт:** <https://d2718281828nis.github.io/ML-overlay-problem/>
(`?lang=ru` / `?lang=en`, `?data=orig` / `dist` / `custom`)

---

<a id="english"></a>

## English

### What this repository is about

Every row of `df_red_green_overlay.csv` is a 2-D point (X, Y) with a colour: 1,445 green and 227 red (13.6%). The task is to recover the colour from the coordinates alone. The repository collects many attempts (clustering, density, embeddings, graphs, supervised classifiers), measures them honestly, and explains why most of them fail on the original data.

Main findings:

- **The baseline trap.** Labelling *every* point red gives F1 = 2p / (1 + p) = **0.239**. About half of the notebook's results sit within ±0.02 of this number, so they have learned nothing.
- **In the original data the colours share one distribution.** The class means and spreads differ by only 1–7% of a standard deviation, so global methods (K-Means, GMM, Fuzzy C-Means, spectral clustering) cut the plane geometrically, not by colour.
- **The only signal is local sparsity.** Red points sit in slightly emptier spots. The plain distance to the k-th nearest neighbour reaches F1 0.39 (k = 10) to 0.46 (k = 5) without labels, as good as the supervised SVM (0.41).
- **When the distributions really differ, the matching model wins.** In the *distinct* dataset (red on a noisy circle, green uniform), a "ring + uniform background" model fitted by EM gets 0.72 without labels, close to the Bayes ceiling of 0.79.
- **t-SNE and graph propagation do not create information.** t-SNE evens out density, and k-NN graph propagation averages red points with their green neighbours. Both help only in small doses.

### Datasets

| File | How it was made | Points |
|---|---|---|
| `df_red_green_overlay.csv` | Original data (columns X, Y, Color, Type; Type is redundant: cross = green, point = red) | 1,672 (227 red) |
| `df_red_green_distinct.csv` | `make_distinct_dataset.py` with default options | 1,704 (227 red) |
| `df_red_green_chains.csv` | `make_distinct_dataset.py --green chains --red chains`: the example where an unsupervised graph method wins (see below) | 1,672 (227 red) |
| `df_red_green_circles.csv` | `... --green chains --red chains --chain-shape circle --seed 5`: the same idea with chains shaped as closed circles | 1,671 (227 red) |

How the distinct dataset is generated:

- **Footprint F:** every location within 45 px of an original point, which keeps the original shape, including its holes.
- **Green:** uniform inside F (a homogeneous Poisson process). The count is drawn from Poisson(1445).
- **Red:** a noisy circle. The angle θ is uniform, and the radius r ~ N(R = 350, σ = 15) px around the centre (1046, 904). Red is *not* restricted to F (17 of 227 points fall outside), unless you pass `--clip-red`.
- **Best possible F1** (the Bayes optimum, using the true generator): ≈ 0.79.

Make your own variant:

```bash
python make_distinct_dataset.py --red gaussian --sigma 120 --center 800 700 --out my_data.csv
python make_distinct_dataset.py --green gaussian --green-sigma 400 --red ring --radius 250 --sigma 25
python make_distinct_dataset.py --help        # all options
```

Options: `--green uniform|gaussian|chains`, `--n-green`, `--green-sigma`, `--green-center X Y`, `--red ring|gaussian|uniform|chains`, `--n-red`, `--red-chains`, `--chain-len MIN MAX`, `--step`, `--gap`, `--turn`, `--chain-shape walk|circle`, `--center X Y`, `--radius`, `--sigma`, `--clip-red`, `--seed`, `--source`, `--out`. Run with no options, the script reproduces the committed file byte for byte.

### The interactive site (`docs/index.html`)

#### How it works

- **One self-contained file.** No server, no build step and no external libraries at runtime. Both CSVs and the precomputed scikit-learn t-SNE maps are embedded in the page.
- **Every method runs in your browser,** implemented in plain JavaScript, on all points of the selected dataset. The results agree with scikit-learn: the reference implementations in `overlay_clustering_advanced.py` give the same F1 to about ±0.02.
- **t-SNE** for custom data and other perplexities is computed live: exact t-SNE with scikit-learn's defaults, in a background Web Worker, about 2 s for 1,700 points. The map animates while it converges.
- **Supervised methods** are scored out of sample, with 5-fold cross-validation or leave-one-out, so a point is never predicted by a model that saw its colour.
- **Bilingual (EN/RU)** with a switch in the top bar; the choice is remembered. It has light and dark themes and works on phones.
- **URL parameters:** `?lang=en|ru` sets the language. `?data=orig|dist|custom` sets the dataset, and the address bar updates when you switch.

#### Page sections

| Section | What you can do |
|---|---|
| **Dataset switch** (hero card and top bar) | Choose *Original*, *Distinct* or *Custom*. Every section follows the switch. |
| **The problem & dataset designer** | Summary numbers and class statistics for the current dataset. The **designer**: pick green (uniform / Gaussian) and red (ring / Gaussian / uniform) distributions, counts, radius, width and seed, then **click the map to place the red centre**. The data regenerates and every classifier reruns; the results table on the right ranks them. The equivalent `make_distinct_dataset.py` command is shown under the map. |
| **Interactive lab** | Two tabs: **Unsupervised learning** (sees only X, Y) and **Supervised learning** (uses the true colours). Each method has a one-line description and parameter sliders. The plot shows the outcome: marker shape is the true class (● red, × green), and a **yellow diamond ◇ marks every wrong prediction**. Views: Outcome / Truth / Score, and X–Y or **t-SNE map**. Below the plot: F1, precision, recall, gain over a random guess, the confusion matrix, and an **F1 vs. fraction flagged** curve (drag it to set how many points to flag). Graph methods also draw the k-NN graph edges. |
| **Baseline trap** | A slider showing the F1 that random guessing gets for any fraction of flagged points. |
| **Where the signal is** | Distance to the k-th neighbour and share of red neighbours, by class, with a k slider. |
| **Distinct data** | How the distinct dataset was created (formulas), its density profile across the circle, a comparison table of all methods on all datasets, and why t-SNE works poorly. |
| **Notebook results** | F1 bars from the notebook (original outputs, or the rerun on the distinct data), with the baseline line. |
| **Repository map / Takeaways** | What each file does, what was fixed, and the main lessons. |

The lab and the designer are synchronised:

- Changing any designer setting, or clicking its map, switches the whole page (lab included) to your custom data.
- The method selected in the lab is highlighted in the designer table.
- Clicking a table row opens that method in the lab.

#### Methods on the site

**Unsupervised (no labels):**

| Group | Methods |
|---|---|
| Reference | all-red baseline |
| Global clustering | K-Means, 2-component GMM, **ring + uniform mixture (EM)**, Gaussian + uniform mixture (EM), Fuzzy C-Means |
| Density | DBSCAN, k-NN density, k-NN sparsity, LNND, LOF |
| Embedding | t-SNE + k-NN density |
| Graphs | **connected components** of a small-k k-NN graph, k-NN graph with GCN-style smoothed density, spectral cut of the graph |

**Supervised (uses the true colours):**

| Group | Methods |
|---|---|
| Reference | Bayes optimum, which knows the true generator: a theoretical limit, not a real method |
| Classifiers | polynomial logistic regression (sliders for degree and C), QDA, k-NN classifier |
| Graph learning | label propagation, SGC (simplified graph convolutional network), **hyperbolic GCN** (Poincaré ball, trained in a background worker) |
| Clustering + labels | t-SNE + DBSCAN, where the true colours pick the red cluster |

"Uses labels" means the method saw the true colours, whether to train, to choose a cluster, or to know the generator. Its score is therefore not comparable with the unsupervised methods. The graph methods are adapted from [ML-MachineLearning-Graphs](https://github.com/D2718281828nis/ML-MachineLearning-Graphs), without that repository's "type-based edges": they connect points of the same Type, which is the colour, so they leak the answer.

#### Key results (F1, default parameters)

| Method | Labels? | Original | Distinct (ring) |
|---|---|---|---|
| All red (baseline) | no | 0.239 | 0.235 |
| K-Means / GMM | no | 0.22 / 0.14 | 0.23 / 0.06 |
| k-NN sparsity | no | **0.39** | 0.00 |
| k-NN density | no | 0.04 | 0.62 |
| Graph: smoothed density | no | 0.05 | 0.64 |
| Ring + uniform mixture (EM) | no | 0.07 | **0.72** |
| t-SNE + k-NN density | no | 0.08 | 0.31 |
| t-SNE + DBSCAN | yes | 0.18 | 0.39 |
| Logistic regression (degree 4, C = 10⁴) | yes | 0.41 | **0.75** |
| Graph: SGC | yes | 0.41–0.44 | 0.66 |
| k-NN classifier | yes | 0.41 | 0.63 |
| Bayes optimum | yes (generator) | – | 0.79 |

#### When does an unsupervised graph method win? (`df_red_green_chains.csv`)

A graph only sees X, Y, so "connectivity" must show up in the geometry as gaps along a structure. The example satisfies four conditions:

1. **Overlay:** red is spread over the same area as green. The class centres differ by 0.02–0.05σ and the spreads by 10–16%.
2. **The same local density:** every point lies on a chain with a 12 px step, so the median distance to the 2nd neighbour is 12.2 px for both colours.
3. **The same local shape:** every point is "a piece of a chain".
4. **Only global connectivity differs:** red = 6 long chains (~38 points each), green ≈ 95 short chains (12–20 points). Different chains stay ≥ 30 px apart.

| Method | F1 |
|---|---|
| **Graph: connected components (k-NN graph, k = 2), no labels** | **1.000** |
| every other unsupervised method (density, LOF, K-Means, GMM, EM models, t-SNE, spectral cut, Louvain, DBSCAN) | 0.03–0.25 (baseline 0.239) |
| supervised k-NN classifier / label propagation / SGC / logistic regression | 0.90 / 0.83 / 0.45 / 0.43 |

Across seeds 1–5 the graph method gets 1.000 in four runs and 0.912 in one. k matters: a chain point has 2 neighbours, so k = 2 keeps chains apart, while k ≥ 3 adds edges between chains and the components merge. On the site, press **"Graph example: chains"** in the designer.

#### Variant: chains shaped as circles (`df_red_green_circles.csv`)

`python make_distinct_dataset.py --green chains --red chains --chain-shape circle --seed 5`. Red = 6 closed circles of ~38 points (radius ≈ 73 px), green ≈ 90 small circles of 12–20 points (radius 23–38 px). The step (12 px) and the gap between circles (≥ 30 px) are the same as for the walk chains.

| Seed | Overlay (max \|Δmean\|/σ) | Graph: connected components | Best non-graph unsupervised (k-NN sparsity) | Supervised k-NN |
|---|---|---|---|---|
| 1 | 0.51 | **1.000** | 0.643 | 0.991 |
| 2 | 0.38 | **1.000** | 0.564 | 0.996 |
| 3 | 0.40 | **1.000** | 0.811 | 1.000 |
| 4 | 0.70 | **1.000** | 0.696 | 1.000 |
| 5 (saved file) | **0.19** | **1.000** | 0.753 | 1.000 |
| 42 | 0.67 | **1.000** | 0.599 | 0.991 |

- **Cleaner for the graph:** closed loops have no loose ends, so connected components work at k = 2 *and* k = 3, and Louvain communities also reach 1.000.
- **Less strict as a test:** a small circle keeps all its points within one diameter (46–76 px), while a large circle is flatter. Local statistics at k ≥ 5 therefore differ, so k-NN sparsity reaches 0.47–0.81 and supervised k-NN about 1.0. The walk chains remain the stricter example, where *only* connectivity differs.
- **The overlay depends on the seed** (0.19–0.70σ), because large circles fit only in the wide parts of the area. Seed 5 has the best overlay. The generator now gives up with a warning when the area is full, instead of hanging.

#### Why the graph works so well here

Illustrated step by step, with seven figures: **[illustrations/README.md](illustrations/README.md)**.

1. **The colour is a property of the whole chain, not of a neighbourhood.** Locally every point looks the same: two neighbours 12 px away and nothing else within 30 px. Any method that looks at a fixed neighbourhood sees the same picture for red and green: k-NN density, LOF, K-Means, a classifier on X, Y, even supervised k-NN with 15 neighbours. The difference (38 points in the chain vs. 12–20) only exists at the scale of the whole chain.
2. **With k = 2 the graph reproduces the true structure exactly.** The step (12 px) is smaller than the gap between chains (30 px), so a point's two nearest neighbours are always its own chain neighbours. 100% of the edges connect points of the same chain, compared with 60% on the ring dataset and 32% on the original data. The graph's connected components are therefore exactly the chains.
3. **Connected components carry information arbitrarily far without averaging it away.** Following edges is transitive: a point in the middle of a 38-point chain is linked to all 38 points, however far away they are. GCN-style averaging (smoothed density, SGC) instead mixes values and fades after a few hops. That is why the component search gets 1.000, while label propagation (supervised) gets 0.83 and SGC 0.45.
4. **Only the right k works.** A chain point has exactly 2 neighbours, so k = 2 matches the structure. At k ≥ 3, extra edges jump to other chains and the components merge (walk chains: k = 3 → 0.28, k = 5 → 0.12). Closed circles tolerate k = 3 because the extra edges stay inside the circle.
5. **The general rule from all experiments:** a graph method beats local methods only when (a) the edges are *homophilous* (neighbours share the class) and (b) the class depends on *long-range structure* (paths, components, communities) rather than on local density. On the original and ring data (a) fails, so graphs at best match k-NN. On the chains both hold, so the graph wins outright.

#### Running and publishing

- **Locally:** open `docs/index.html` in a browser. It works from `file://` too, including the t-SNE worker.
- **GitHub Pages:** *Settings → Pages → Branch `main`, folder `/docs`*. After a push, hard-refresh the page (Cmd/Ctrl + Shift + R), because Pages caches pages for up to 10 minutes.
- **The data is embedded in the page.** Regenerating a CSV does not update the site; the page has to be rebuilt with the new data.

### Repository contents

| File | Purpose |
|---|---|
| `docs/index.html` | The interactive site described above |
| `case_study-unsupervised-overlay-problem.ipynb` | Main study (Colab-friendly). It runs top to bottom on `df_red_green_distinct.csv` and has saved outputs. To use the original data, change `DATA_FILE` in the load cell. |
| `overlay_clustering_advanced.py` | Nine method families with F1, precision, recall, figures and a summary table. Includes **t-SNE** (method 8) and **graph methods** (method 9, with connected components as 9e). |
| `overlay_advanced_tuning.py` | GMM BIC/AIC, spectral gamma search, LOF weighting, multi-metric spectral clustering, bootstrap consensus. Runs in about 8 s. |
| `quickstart_colab.py` | Copy-paste Colab template: GMM + threshold search. The threshold is tuned on true labels, so it is supervised. |
| `make_distinct_dataset.py` | Generator for datasets with chosen red and green distributions |
| `hyperbolic_gcn.py` | NumPy hyperbolic GCN (`HyperbolicGCN` from ML-MachineLearning-Graphs, Poincaré ball), used as method 9g of `overlay_clustering_advanced.py` |
| `illustrations/` | Seven figures and an EN/RU explanation of why the graph method wins on the chain datasets (`make_illustrations.py` regenerates them) |
| `df_red_green_overlay.csv`, `df_red_green_distinct.csv`, `df_red_green_chains.csv`, `df_red_green_circles.csv` | Datasets |

Run the scripts on any dataset:

```bash
python overlay_clustering_advanced.py                                     # distinct data (default)
OVERLAY_DATA=df_red_green_overlay.csv python overlay_clustering_advanced.py
OVERLAY_DATA=my_data.csv python overlay_advanced_tuning.py
```

Requirements: Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn. The notebook also uses networkx, torch and tqdm.

<a id="history"></a>

### History of the experiments

How the project got here, step by step. All numbers are F1 for the red class.

| # | Step | What we found |
|---|---|---|
| 1 | Original notebook: 15+ methods (DBSCAN, agglomerative, FCM, t-SNE, spectral, ICA, KDE, hyperbolic graph/GCN, LNND, OCSVM, supervised LogReg/RF/SVM) | Most scores 0.22–0.26, and **0.2391 is exactly the F1 of "everything red"** (2p/(1+p)). Best label-free: LNND 0.295. Supervised SVM: 0.41. |
| 2 | Analysis of the data | Red and green share one distribution (means differ by 1–7% of σ). The only signal is local sparsity: the plain k-NN distance gives 0.39 (k = 10) to 0.46 (k = 5) without labels. |
| 3 | Audit of the code | Fixed a spectral step that hung, a 145M-iteration loop, an LOF that was never fitted, `fit_predict_proba`, notebook cell order, `TSNE(n_iter)`, `get_cmap`, an unseeded GCN, and unstable cluster→colour orientation. |
| 4 | Distinct dataset v1: red Gaussian blob (σ = 90 px) on uniform green | Density works: k-NN density 0.73, and the Gaussian + uniform EM reaches the Bayes level of 0.74–0.77. t-SNE + DBSCAN reaches 0.64 only when labels choose the cluster. |
| 5 | Distinct v2: red noisy ring (R = 350, σ = 15) | Ring + uniform EM 0.72 without labels (Bayes 0.79); k-NN density 0.62; t-SNE + DBSCAN 0.39. |
| 6 | Bug fix: red was forced inside the green area | A ring near the edge collapsed into the overlap. Red now follows its own distribution; `--clip-red` restores the old behaviour. |
| 7 | Interactive site: bilingual, with the dataset designer | Every method runs live in the browser; the designer and the lab are synchronised. |
| 8 | t-SNE as a classifier (exact t-SNE in a Web Worker) | Weak: t-SNE evens out density (k-NN density on the t-SNE map 0.31 vs. 0.62 on X–Y), and a lower perplexity is even worse. |
| 9 | Split into supervised / unsupervised; logistic regression and QDA added | Polynomial logistic regression reaches 0.75 on the ring, but only with weak regularisation (C = 1 gives 0.07). QDA gets 0.16, because it is the wrong model. |
| 10 | Graph methods from [ML-MachineLearning-Graphs](https://github.com/D2718281828nis/ML-MachineLearning-Graphs) | Smoothed density 0.64, label propagation 0.63, SGC 0.66, spectral cut 0.18. The repository's "type-based edges" were removed, because they leak the colour. |
| 11 | Why graphs ≈ k-NN on these datasets | Edges mix the colours (only 60% / 32% of a red node's edges lead to red), so propagation averages red with green. SGC without the graph: 0.744; with 4 hops: 0.648. |
| 12 | An example where graphs win: chains (random walks) | Connected components 1.000; every other unsupervised method ≤ 0.25; supervised k-NN 0.83. |
| 13 | The same with circles | Connected components 1.000 on every seed, but small circles leak a local signal: k-NN sparsity 0.47–0.81, supervised k-NN ≈ 1.0. |
| 14 | Hyperbolic GCN from ML-MachineLearning-Graphs (`HyperbolicGCN`: Poincaré-ball lift, tangent-space GCN) | The repository's notebook ran it on the original data without class weights and found no red point (recall 0). With a class-balanced loss and 5-fold CV: original 0.40, ring 0.68–0.76 (depends on curvature c), walk chains 0.75. In this architecture logmap0(expmap0(v)) = v except where the ball's edge clips v, so the curvature only bounds the hidden-vector length (c → 0 is a plain GCN). |

### Lessons

1. Always report the trivial baseline (all red: F1 = 2p / (1 + p)).
2. Global-blob methods cannot separate classes that share one distribution.
3. Model the actual difference between the classes (density, shape) directly.
4. Use t-SNE to *look* at data, not to classify it.
5. Watch for label leakage: cluster-to-colour mapping, hyper-parameters tuned by F1 on true labels, Type-based graph edges.
6. Compare score-based methods with the F1-vs-fraction (precision–recall) curve, not a single threshold.

---

<a id="русский"></a>

## Русский

### О чём этот репозиторий

Каждая строка `df_red_green_overlay.csv` — точка на плоскости (X, Y) и её цвет: 1 445 зелёных и 227 красных (13,6%). Нужно восстановить цвет только по координатам. В репозитории собрано много попыток (кластеризация, плотность, вложения, графы, классификаторы с учителем). Они честно измерены, и объяснено, почему большинство из них не работает на исходных данных.

Главные выводы:

- **Ловушка базового уровня.** Если пометить красными *все* точки, F1 = 2p / (1 + p) = **0,239**. Около половины результатов ноутбука лежат в пределах ±0,02 от этого числа, то есть ничему не научились.
- **В исходных данных у цветов одно и то же распределение.** Средние и разброс классов отличаются лишь на 1–7% стандартного отклонения, поэтому глобальные методы (K-Means, GMM, Fuzzy C-Means, спектральная кластеризация) режут плоскость геометрически, а не по цвету.
- **Единственный сигнал — локальная разреженность.** Красные точки стоят в чуть более пустых местах. Простое расстояние до k-го ближайшего соседа даёт без меток F1 от 0,39 (k = 10) до 0,46 (k = 5) — столько же, сколько SVM с учителем (0,41).
- **Когда распределения действительно разные, побеждает подходящая модель.** На *различимом* датасете (красные — на зашумлённой окружности, зелёные — равномерно) модель «кольцо + равномерный фон», подогнанная EM, даёт без меток 0,72 при байесовском потолке 0,79.
- **t-SNE и распространение по графу не создают информацию.** t-SNE выравнивает плотность, а распространение по графу k-NN усредняет красные точки с их зелёными соседями. Оба помогают только в малой дозе.

### Датасеты

| Файл | Как получен | Точек |
|---|---|---|
| `df_red_green_overlay.csv` | Исходные данные (столбцы X, Y, Color, Type; Type избыточен: cross = зелёный, point = красный) | 1 672 (227 красных) |
| `df_red_green_distinct.csv` | `make_distinct_dataset.py` с параметрами по умолчанию | 1 704 (227 красных) |
| `df_red_green_chains.csv` | `make_distinct_dataset.py --green chains --red chains`: пример, где побеждает метод на графе без учителя (см. ниже) | 1 672 (227 красных) |
| `df_red_green_circles.csv` | `... --green chains --red chains --chain-shape circle --seed 5`: та же идея, но цепочки — замкнутые окружности | 1 671 (227 красных) |

Как создаётся различимый датасет:

- **Область F:** все места в пределах 45 px от точек исходного файла. Так сохраняется исходная форма, включая «дыры».
- **Зелёные:** равномерно внутри F (однородный пуассоновский процесс). Количество берётся из Poisson(1445).
- **Красные:** зашумлённая окружность. Угол θ равномерный, радиус r ~ N(R = 350, σ = 15) px вокруг центра (1046, 904). Красные *не* ограничены областью F (17 из 227 точек оказываются вне её), если не указать `--clip-red`.
- **Лучший возможный F1** (байесовский оптимум, знающий истинный генератор): ≈ 0,79.

Свой вариант:

```bash
python make_distinct_dataset.py --red gaussian --sigma 120 --center 800 700 --out my_data.csv
python make_distinct_dataset.py --green gaussian --green-sigma 400 --red ring --radius 250 --sigma 25
python make_distinct_dataset.py --help        # все параметры
```

Без параметров скрипт воспроизводит файл из репозитория байт в байт.

### Интерактивный сайт (`docs/index.html`)

#### Как он устроен

- **Один самодостаточный файл.** Без сервера, без сборки и без внешних библиотек во время работы. Оба CSV и заранее посчитанные в scikit-learn карты t-SNE встроены в страницу.
- **Все методы выполняются в вашем браузере,** на чистом JavaScript, на всех точках выбранного датасета. Результаты совпадают со scikit-learn: эталонные реализации в `overlay_clustering_advanced.py` дают те же F1 с точностью примерно ±0,02.
- **t-SNE** для своего датасета и других perplexity считается на месте: точный t-SNE с параметрами scikit-learn, в фоновом Web Worker, около 2 с на 1 700 точек. Карта анимируется, пока сходится.
- **Методы с учителем** оцениваются на отложенных точках (5-кратная кросс-валидация или leave-one-out), поэтому точку никогда не предсказывает модель, видевшая её цвет.
- **Два языка (EN/RU)** с переключателем вверху; выбор запоминается. Есть светлая и тёмная темы, страница работает на телефоне.
- **Параметры адреса:** `?lang=en|ru` задаёт язык. `?data=orig|dist|custom` задаёт датасет; адресная строка обновляется при переключении.

#### Разделы страницы

| Раздел | Что можно делать |
|---|---|
| **Переключатель датасета** (карточка вверху и верхняя панель) | Выбрать *Исходный*, *Различимый* или *Свой*. Все разделы следуют переключателю. |
| **Задача и конструктор датасета** | Итоговые числа и статистика классов текущего датасета. **Конструктор**: выберите распределения зелёных (равномерное / гауссово) и красных (кольцо / гауссово / равномерное), количество, радиус, ширину и зерно, затем **щёлкните по карте, чтобы задать центр красных**. Данные пересоздаются, все классификаторы перезапускаются, и таблица справа их ранжирует. Под картой показана такая же команда `make_distinct_dataset.py`. |
| **Интерактивная лаборатория** | Две вкладки: **Обучение без учителя** (видит только X, Y) и **Обучение с учителем** (использует истинные цвета). У каждого метода есть описание в одну строку и ползунки параметров. На графике форма маркера — истинный класс (● красный, × зелёный), а **жёлтый ромб ◇ отмечает каждую ошибку**. Режимы: Результат / Истина / Оценка, пространство X–Y или **карта t-SNE**. Под графиком: F1, точность, полнота, выигрыш над случайным угадыванием, матрица ошибок и **кривая F1 от доли отмеченных** (перетащите, чтобы выбрать, сколько точек отмечать). Для методов на графах рисуются рёбра графа k-NN. |
| **Ловушка базового уровня** | Ползунок: какой F1 даёт случайное угадывание при любой доле отмеченных точек. |
| **Где сигнал** | Расстояние до k-го соседа и доля красных среди соседей по классам, с ползунком k. |
| **Различимые данные** | Как создан различимый датасет (формулы), профиль плотности поперёк окружности, таблица всех методов на всех датасетах и объяснение, почему t-SNE работает плохо. |
| **Результаты ноутбука** | Столбцы F1 из ноутбука (исходные результаты или перезапуск на различимых данных) с линией базового уровня. |
| **Карта репозитория / Выводы** | Что делает каждый файл, что исправлено, и главные уроки. |

Лаборатория и конструктор синхронизированы:

- Любое изменение в конструкторе или щелчок по его карте переключает всю страницу (и лабораторию) на ваш датасет.
- Метод, выбранный в лаборатории, подсвечивается в таблице конструктора.
- Щелчок по строке таблицы открывает этот метод в лаборатории.

#### Методы на сайте

**Без учителя (без меток):**

| Группа | Методы |
|---|---|
| Эталон | все красные (базовый уровень) |
| Глобальная кластеризация | K-Means, GMM из 2 компонент, **смесь «кольцо + фон» (EM)**, смесь «гаусс + фон» (EM), Fuzzy C-Means |
| Плотность | DBSCAN, плотность k-NN, разреженность k-NN, LNND, LOF |
| Вложение | t-SNE + плотность k-NN |
| Графы | **компоненты связности** графа k-NN с малым k, граф k-NN со сглаженной (как в GCN) плотностью, спектральный разрез графа |

**С учителем (использует истинные цвета):**

| Группа | Методы |
|---|---|
| Эталон | байесовский оптимум, знающий истинный генератор: теоретический предел, а не реальный метод |
| Классификаторы | полиномиальная логистическая регрессия (ползунки степени и C), QDA, классификатор k-NN |
| Обучение на графе | распространение меток, SGC (упрощённая графовая свёрточная сеть), **гиперболическая GCN** (шар Пуанкаре, обучается в фоновом потоке) |
| Кластеризация + метки | t-SNE + DBSCAN, где красный кластер выбирают истинные цвета |

«Использует метки» означает, что метод видел истинные цвета: при обучении, при выборе кластера или как знание генератора. Поэтому его результат нельзя напрямую сравнивать с методами без учителя. Методы на графах адаптированы из [ML-MachineLearning-Graphs](https://github.com/D2718281828nis/ML-MachineLearning-Graphs), но без его «рёбер по Type»: они соединяют точки одного Type, а Type — это цвет, то есть такие рёбра выдают ответ.

#### Главные результаты (F1, параметры по умолчанию)

| Метод | Метки? | Исходный | Различимый (кольцо) |
|---|---|---|---|
| Все красные (базовый) | нет | 0,239 | 0,235 |
| K-Means / GMM | нет | 0,22 / 0,14 | 0,23 / 0,06 |
| Разреженность k-NN | нет | **0,39** | 0,00 |
| Плотность k-NN | нет | 0,04 | 0,62 |
| Граф: сглаженная плотность | нет | 0,05 | 0,64 |
| Смесь «кольцо + фон» (EM) | нет | 0,07 | **0,72** |
| t-SNE + плотность k-NN | нет | 0,08 | 0,31 |
| t-SNE + DBSCAN | да | 0,18 | 0,39 |
| Логистическая регрессия (степень 4, C = 10⁴) | да | 0,41 | **0,75** |
| Граф: SGC | да | 0,41–0,44 | 0,66 |
| Классификатор k-NN | да | 0,41 | 0,63 |
| Байесовский оптимум | да (генератор) | – | 0,79 |

#### Когда побеждает метод на графе без учителя? (`df_red_green_chains.csv`)

Граф видит только X, Y, поэтому «связность» должна проявляться в геометрии — как промежутки вдоль структуры. Пример выполняет четыре условия:

1. **Наложение:** красные распределены по той же области, что и зелёные. Центры классов отличаются на 0,02–0,05σ, разброс — на 10–16%.
2. **Одинаковая локальная плотность:** каждая точка лежит на цепочке с шагом 12 px, медиана расстояния до 2-го соседа — 12,2 px у обоих цветов.
3. **Одинаковая локальная форма:** каждая точка — «кусочек цепочки».
4. **Отличается только глобальная связность:** красные — 6 длинных цепочек (~38 точек), зелёные — около 95 коротких (12–20 точек). Разные цепочки держатся не ближе 30 px.

| Метод | F1 |
|---|---|
| **Граф: компоненты связности (граф k-NN, k = 2), без меток** | **1,000** |
| все остальные методы без учителя (плотность, LOF, K-Means, GMM, модели EM, t-SNE, спектральный разрез, Louvain, DBSCAN) | 0,03–0,25 (базовый 0,239) |
| с учителем: классификатор k-NN / распространение меток / SGC / логистическая регрессия | 0,90 / 0,83 / 0,45 / 0,43 |

На зёрнах 1–5 метод на графе даёт 1,000 в четырёх запусках и 0,912 в одном. Важно k: у точки цепочки 2 соседа, поэтому k = 2 не соединяет цепочки, а при k ≥ 3 появляются рёбра между цепочками и компоненты сливаются. На сайте нажмите **«Пример для графа: цепочки»** в конструкторе.

#### Вариант: цепочки в форме окружностей (`df_red_green_circles.csv`)

`python make_distinct_dataset.py --green chains --red chains --chain-shape circle --seed 5`. Красные — 6 замкнутых окружностей по ~38 точек (радиус ≈ 73 px), зелёные — около 90 маленьких окружностей по 12–20 точек (радиус 23–38 px). Шаг (12 px) и промежуток между окружностями (≥ 30 px) те же, что у цепочек-блужданий.

| Зерно | Наложение (max \|Δсреднего\|/σ) | Граф: компоненты связности | Лучший метод без учителя не на графе (разреженность k-NN) | k-NN с учителем |
|---|---|---|---|---|
| 1 | 0,51 | **1,000** | 0,643 | 0,991 |
| 2 | 0,38 | **1,000** | 0,564 | 0,996 |
| 3 | 0,40 | **1,000** | 0,811 | 1,000 |
| 4 | 0,70 | **1,000** | 0,696 | 1,000 |
| 5 (сохранённый файл) | **0,19** | **1,000** | 0,753 | 1,000 |
| 42 | 0,67 | **1,000** | 0,599 | 0,991 |

- **Для графа ещё проще:** у замкнутой петли нет свободных концов, поэтому компоненты связности работают и при k = 2, *и* при k = 3, а сообщества Louvain тоже дают 1,000.
- **Как тест — менее строгий:** маленькая окружность держит все свои точки в пределах одного диаметра (46–76 px), а большая более плоская. Локальная статистика при k ≥ 5 поэтому отличается: разреженность k-NN даёт 0,47–0,81, k-NN с учителем — около 1,0. Цепочки-блуждания остаются более строгим примером, где отличается *только* связность.
- **Наложение зависит от зерна** (0,19–0,70σ): большие окружности помещаются только в широких частях области. У зерна 5 наложение лучшее. Генератор теперь останавливается с предупреждением, когда область заполнена, а не зависает.

#### Почему граф здесь работает так хорошо

Пошагово, с семью рисунками: **[illustrations/README.md](illustrations/README.md#русский)**.

1. **Цвет — свойство всей цепочки, а не окрестности.** Локально все точки выглядят одинаково: два соседа в 12 px и больше ничего в радиусе 30 px. Любой метод, смотрящий на фиксированную окрестность, видит для красных и зелёных одну и ту же картину: плотность k-NN, LOF, K-Means, классификатор на X, Y, даже k-NN с учителем по 15 соседям. Разница (38 точек в цепочке против 12–20) существует только в масштабе всей цепочки.
2. **При k = 2 граф точно воспроизводит истинную структуру.** Шаг (12 px) меньше промежутка между цепочками (30 px), поэтому два ближайших соседа точки — всегда её соседи по цепочке. 100% рёбер соединяют точки одной цепочки, против 60% на датасете с кольцом и 32% на исходных данных. Компоненты связности графа поэтому в точности совпадают с цепочками.
3. **Компоненты связности переносят информацию сколь угодно далеко, не усредняя её.** Переход по рёбрам транзитивен: точка в середине цепочки из 38 точек связана со всеми 38, как бы далеко они ни были. Усреднение в духе GCN (сглаженная плотность, SGC), наоборот, смешивает значения и затухает через несколько шагов. Поэтому поиск компонент даёт 1,000, а распространение меток (с учителем) — 0,83 и SGC — 0,45.
4. **Работает только правильное k.** У точки цепочки ровно 2 соседа, поэтому k = 2 совпадает со структурой. При k ≥ 3 лишние рёбра перескакивают на другие цепочки, и компоненты сливаются (цепочки-блуждания: k = 3 → 0,28, k = 5 → 0,12). Замкнутые окружности выдерживают k = 3, потому что лишние рёбра остаются внутри окружности.
5. **Общее правило по всем экспериментам:** метод на графе обгоняет локальные методы, только когда (а) рёбра *гомофильны* (соседи одного класса) и (б) класс определяется *дальней структурой* (пути, компоненты, сообщества), а не локальной плотностью. На исходных данных и на кольце не выполняется (а), поэтому графы в лучшем случае равны k-NN. На цепочках выполняются оба условия, и граф выигрывает безоговорочно.

#### Запуск и публикация

- **Локально:** откройте `docs/index.html` в браузере. Работает и с `file://`, включая t-SNE в фоновом потоке.
- **GitHub Pages:** *Settings → Pages → Branch `main`, папка `/docs`*. После push обновите страницу без кэша (Cmd/Ctrl + Shift + R): Pages кэширует страницы до 10 минут.
- **Данные встроены в страницу.** Пересоздание CSV не обновляет сайт; страницу нужно пересобрать с новыми данными.

### Состав репозитория

| Файл | Назначение |
|---|---|
| `docs/index.html` | Интерактивный сайт, описанный выше |
| `case_study-unsupervised-overlay-problem.ipynb` | Основное исследование (для Colab). Выполняется целиком на `df_red_green_distinct.csv`, результаты сохранены. Для исходных данных поменяйте `DATA_FILE` в ячейке загрузки. |
| `overlay_clustering_advanced.py` | Девять семейств методов с F1, точностью, полнотой, графиками и итоговой таблицей. Включает **t-SNE** (метод 8) и **методы на графах** (метод 9, компоненты связности — 9e). |
| `overlay_advanced_tuning.py` | BIC/AIC для GMM, подбор gamma для спектральной кластеризации, веса LOF, спектральная кластеризация с разными метриками, бутстреп-консенсус. Работает около 8 с. |
| `quickstart_colab.py` | Шаблон для Colab: GMM + подбор порога. Порог подбирается по истинным меткам, то есть это уже обучение с учителем. |
| `make_distinct_dataset.py` | Генератор датасетов с выбранными распределениями красных и зелёных |
| `hyperbolic_gcn.py` | Гиперболическая GCN на NumPy (`HyperbolicGCN` из ML-MachineLearning-Graphs, шар Пуанкаре), метод 9g в `overlay_clustering_advanced.py` |
| `illustrations/` | Семь рисунков и объяснение на EN/RU, почему метод на графе побеждает на датасетах с цепочками (`make_illustrations.py` пересоздаёт их) |
| `df_red_green_overlay.csv`, `df_red_green_distinct.csv`, `df_red_green_chains.csv`, `df_red_green_circles.csv` | Датасеты |

Запуск скриптов на любом датасете:

```bash
python overlay_clustering_advanced.py                                     # различимые данные (по умолчанию)
OVERLAY_DATA=df_red_green_overlay.csv python overlay_clustering_advanced.py
OVERLAY_DATA=my_data.csv python overlay_advanced_tuning.py
```

Нужно: Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn. Ноутбуку также нужны networkx, torch и tqdm.

<a id="история"></a>

### История экспериментов

Как проект пришёл к текущему состоянию, шаг за шагом. Все числа — F1 для красного класса.

| № | Шаг | Что выяснилось |
|---|---|---|
| 1 | Исходный ноутбук: 15+ методов (DBSCAN, агломеративная, FCM, t-SNE, спектральная, ICA, KDE, гиперболический граф/GCN, LNND, OCSVM, LogReg/RF/SVM с учителем) | Большинство результатов 0,22–0,26, и **0,2391 — это ровно F1 для «все красные»** (2p/(1+p)). Лучший без меток: LNND 0,295. SVM с учителем: 0,41. |
| 2 | Анализ данных | У красных и зелёных одно распределение (средние отличаются на 1–7% σ). Единственный сигнал — локальная разреженность: простое расстояние k-NN даёт без меток от 0,39 (k = 10) до 0,46 (k = 5). |
| 3 | Аудит кода | Исправлены: зависавший спектральный шаг, цикл на 145 млн итераций, LOF без fit, `fit_predict_proba`, порядок ячеек ноутбука, `TSNE(n_iter)`, `get_cmap`, GCN без зерна, неустойчивое сопоставление «кластер → цвет». |
| 4 | Различимый датасет v1: красное гауссово облако (σ = 90 px) на равномерном зелёном | Плотность работает: плотность k-NN 0,73, а EM «гаусс + фон» достигает байесовского уровня 0,74–0,77. t-SNE + DBSCAN даёт 0,64, только когда кластер выбирают метки. |
| 5 | Различимый v2: красное зашумлённое кольцо (R = 350, σ = 15) | EM «кольцо + фон» без меток 0,72 (байес 0,79); плотность k-NN 0,62; t-SNE + DBSCAN 0,39. |
| 6 | Исправление: красные насильно удерживались внутри зелёной области | Кольцо у края схлопывалось в зону перекрытия. Теперь красные следуют своему распределению; `--clip-red` возвращает старое поведение. |
| 7 | Интерактивный сайт: двуязычный, с конструктором датасета | Все методы выполняются прямо в браузере; конструктор и лаборатория синхронизированы. |
| 8 | t-SNE как классификатор (точный t-SNE в Web Worker) | Слабый: t-SNE выравнивает плотность (плотность k-NN на карте t-SNE 0,31 против 0,62 на X–Y), а меньшая perplexity ещё хуже. |
| 9 | Разделение на «с учителем / без учителя»; добавлены логистическая регрессия и QDA | Полиномиальная логистическая регрессия даёт 0,75 на кольце, но только при слабой регуляризации (при C = 1 — 0,07). QDA — 0,16, потому что это неверная модель. |
| 10 | Методы на графах из [ML-MachineLearning-Graphs](https://github.com/D2718281828nis/ML-MachineLearning-Graphs) | Сглаженная плотность 0,64, распространение меток 0,63, SGC 0,66, спектральный разрез 0,18. «Рёбра по Type» из того репозитория убраны, потому что выдают цвет. |
| 11 | Почему здесь графы ≈ k-NN | Рёбра смешивают цвета (лишь 60% / 32% рёбер красной точки ведут к красной), поэтому распространение усредняет красное с зелёным. SGC без графа: 0,744; с 4 шагами: 0,648. |
| 12 | Пример, где побеждают графы: цепочки (случайные блуждания) | Компоненты связности 1,000; все остальные методы без учителя ≤ 0,25; k-NN с учителем 0,83. |
| 13 | То же с окружностями | Компоненты связности 1,000 на всех зёрнах, но маленькие окружности дают локальный сигнал: разреженность k-NN 0,47–0,81, k-NN с учителем ≈ 1,0. |
| 14 | Гиперболическая GCN из ML-MachineLearning-Graphs (`HyperbolicGCN`: подъём в шар Пуанкаре, GCN в касательном пространстве) | Ноутбук того репозитория запускал её на исходных данных без весов классов и не нашёл ни одной красной точки (полнота 0). Со сбалансированной функцией потерь и 5-кратной CV: исходный датасет 0,40, кольцо 0,68–0,76 (зависит от кривизны c), цепочки-блуждания 0,75. В этой архитектуре logmap0(expmap0(v)) = v везде, кроме мест, где край шара обрезает v, поэтому кривизна лишь ограничивает длину скрытых векторов (c → 0 — обычная GCN). |

### Уроки

1. Всегда приводите тривиальный базовый уровень (все красные: F1 = 2p / (1 + p)).
2. Методы «глобальных облаков» не разделяют классы с одинаковым распределением.
3. Моделируйте реальное различие классов (плотность, форму) напрямую.
4. t-SNE нужен, чтобы *смотреть* на данные, а не чтобы их классифицировать.
5. Следите за утечкой меток: сопоставление «кластер → цвет», гиперпараметры, подобранные по F1 на истинных метках, рёбра графа по Type.
6. Сравнивайте методы с оценкой по кривой F1 от доли отмеченных (precision–recall), а не по одному порогу.
