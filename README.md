# Overlay Problem: separating overlapping red and green points

**[English](#english) · [Русский](#русский)**

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

Options: `--green uniform|gaussian`, `--n-green`, `--green-sigma`, `--green-center X Y`, `--red ring|gaussian|uniform`, `--n-red`, `--center X Y`, `--radius`, `--sigma`, `--clip-red`, `--seed`, `--source`, `--out`. Run with no options, the script reproduces the committed file byte for byte.

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
| Graphs | k-NN graph with GCN-style smoothed density, spectral cut of the graph |

**Supervised (uses the true colours):**

| Group | Methods |
|---|---|
| Reference | Bayes optimum, which knows the true generator: a theoretical limit, not a real method |
| Classifiers | polynomial logistic regression (sliders for degree and C), QDA, k-NN classifier |
| Graph learning | label propagation, SGC (simplified graph convolutional network) |
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

#### Running and publishing

- **Locally:** open `docs/index.html` in a browser. It works from `file://` too, including the t-SNE worker.
- **GitHub Pages:** *Settings → Pages → Branch `main`, folder `/docs`*. After a push, hard-refresh the page (Cmd/Ctrl + Shift + R), because Pages caches pages for up to 10 minutes.
- **The data is embedded in the page.** Regenerating a CSV does not update the site; the page has to be rebuilt with the new data.

### Repository contents

| File | Purpose |
|---|---|
| `docs/index.html` | The interactive site described above |
| `case_study-unsupervised-overlay-problem.ipynb` | Main study (Colab-friendly). It runs top to bottom on `df_red_green_distinct.csv` and has saved outputs. To use the original data, change `DATA_FILE` in the load cell. |
| `overlay_clustering_advanced.py` | Nine method families with F1, precision, recall, figures and a summary table. Includes **t-SNE** (method 8) and **graph methods** (method 9). |
| `overlay_advanced_tuning.py` | GMM BIC/AIC, spectral gamma search, LOF weighting, multi-metric spectral clustering, bootstrap consensus. Runs in about 8 s. |
| `quickstart_colab.py` | Copy-paste Colab template: GMM + threshold search. The threshold is tuned on true labels, so it is supervised. |
| `make_distinct_dataset.py` | Generator for datasets with chosen red and green distributions |
| `df_red_green_overlay.csv`, `df_red_green_distinct.csv` | Datasets |

Run the scripts on any dataset:

```bash
python overlay_clustering_advanced.py                                     # distinct data (default)
OVERLAY_DATA=df_red_green_overlay.csv python overlay_clustering_advanced.py
OVERLAY_DATA=my_data.csv python overlay_advanced_tuning.py
```

Requirements: Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn. The notebook also uses networkx, torch and tqdm.

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
| Графы | граф k-NN со сглаженной (как в GCN) плотностью, спектральный разрез графа |

**С учителем (использует истинные цвета):**

| Группа | Методы |
|---|---|
| Эталон | байесовский оптимум, знающий истинный генератор: теоретический предел, а не реальный метод |
| Классификаторы | полиномиальная логистическая регрессия (ползунки степени и C), QDA, классификатор k-NN |
| Обучение на графе | распространение меток, SGC (упрощённая графовая свёрточная сеть) |
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

#### Запуск и публикация

- **Локально:** откройте `docs/index.html` в браузере. Работает и с `file://`, включая t-SNE в фоновом потоке.
- **GitHub Pages:** *Settings → Pages → Branch `main`, папка `/docs`*. После push обновите страницу без кэша (Cmd/Ctrl + Shift + R): Pages кэширует страницы до 10 минут.
- **Данные встроены в страницу.** Пересоздание CSV не обновляет сайт; страницу нужно пересобрать с новыми данными.

### Состав репозитория

| Файл | Назначение |
|---|---|
| `docs/index.html` | Интерактивный сайт, описанный выше |
| `case_study-unsupervised-overlay-problem.ipynb` | Основное исследование (для Colab). Выполняется целиком на `df_red_green_distinct.csv`, результаты сохранены. Для исходных данных поменяйте `DATA_FILE` в ячейке загрузки. |
| `overlay_clustering_advanced.py` | Девять семейств методов с F1, точностью, полнотой, графиками и итоговой таблицей. Включает **t-SNE** (метод 8) и **методы на графах** (метод 9). |
| `overlay_advanced_tuning.py` | BIC/AIC для GMM, подбор gamma для спектральной кластеризации, веса LOF, спектральная кластеризация с разными метриками, бутстреп-консенсус. Работает около 8 с. |
| `quickstart_colab.py` | Шаблон для Colab: GMM + подбор порога. Порог подбирается по истинным меткам, то есть это уже обучение с учителем. |
| `make_distinct_dataset.py` | Генератор датасетов с выбранными распределениями красных и зелёных |
| `df_red_green_overlay.csv`, `df_red_green_distinct.csv` | Датасеты |

Запуск скриптов на любом датасете:

```bash
python overlay_clustering_advanced.py                                     # различимые данные (по умолчанию)
OVERLAY_DATA=df_red_green_overlay.csv python overlay_clustering_advanced.py
OVERLAY_DATA=my_data.csv python overlay_advanced_tuning.py
```

Нужно: Python 3, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn. Ноутбуку также нужны networkx, torch и tqdm.

### Уроки

1. Всегда приводите тривиальный базовый уровень (все красные: F1 = 2p / (1 + p)).
2. Методы «глобальных облаков» не разделяют классы с одинаковым распределением.
3. Моделируйте реальное различие классов (плотность, форму) напрямую.
4. t-SNE нужен, чтобы *смотреть* на данные, а не чтобы их классифицировать.
5. Следите за утечкой меток: сопоставление «кластер → цвет», гиперпараметры, подобранные по F1 на истинных метках, рёбра графа по Type.
6. Сравнивайте методы с оценкой по кривой F1 от доли отмеченных (precision–recall), а не по одному порогу.
