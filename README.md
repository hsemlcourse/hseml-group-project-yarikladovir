[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — Детекция мошеннических вакансий

**Студент:** Ладовир Ярослав Александрович

**Группа:** БИВ238


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Модели](#модели)
6. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

Цель проекта — построить модель машинного обучения для классификации вакансий на реальные и мошеннические.

Задача формулируется как **бинарная классификация**:

- `0` — реальная вакансия;
- `1` — мошенническая вакансия.

Основные данные — текст вакансий и метаданные.  
В качестве текстовых признаков используются поля:

- `title`;
- `company_profile`;
- `description`;
- `requirements`;
- `benefits`.

На этапе CP1 была построена baseline-модель.  
На этапе CP2 были проведены расширенные NLP-эксперименты: сравнение нескольких моделей, подбор гиперпараметров, дополнительные эксперименты с TF-IDF и анализ ошибок.

**Задача:** бинарная классификация

**Датасет:** Recruitment Scam / EMSCAD

**Источник датасета:** https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam

**Целевая переменная:** `fraudulent`

**Основная метрика:** `F1-score` для fraud-класса

Так как мошеннические вакансии составляют только около 5% датасета, accuracy не является основной метрикой.  
Основное внимание уделяется качеству на классе `fraud`: `precision`, `recall`, `F1-score` и `ROC-AUC`.

## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│ ├── processed # Очищенные и обработанные данные
│ └── raw # Исходные файлы
├── models # Сохранённые модели
├── notebooks
│ ├── 01_eda.ipynb # Анализ данных
│ ├── 02_baseline.ipynb # Baseline-модель
│ └── 03_nlp_experiments.ipynb # NLP-эксперименты tuning и финальная модель
├── presentation # Презентация для защиты
├── report
│ ├── images # Изображения для отчёта
│ ├── experiments.csv # Таблица экспериментов CP2
│ └── report.md # Финальный отчёт
├── src
│ ├── preprocessing.py # Предобработка данных
│ └── modeling.py # Обучение и оценка моделей
├── tests
│ └── test.py # Тесты пайплайна
├── requirements.txt
└── README.md
```

- `notebooks/01_eda.ipynb` — первичный анализ данных и подготовка;
- `notebooks/02_baseline.ipynb` — baseline-модель CP1;
- `notebooks/03_nlp_experiments.ipynb` — основные эксперименты CP2;
- `models/baseline_tfidf_logreg.joblib` — baseline-модель;
- `models/final_tfidf_linear_svc.joblib` — финальная модель CP2;
- `report/experiments.csv` — итоговая таблица экспериментов.


## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить Jupyter Notebook
jupyter notebook
```

## Данные

Используется датасет:

**Recruitment Scam (EMSCAD)**  
Источник: https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam

- Количество объектов: `17 880`
- Количество признаков: `18`
- Таргет: `fraudulent`

Особенности:

- сильный дисбаланс классов: мошеннические вакансии составляют около 5%;
- наличие текстовых, категориальных и бинарных признаков;
- часть текстовых полей содержит HTML-теги;
- часть признаков содержит пропуски.

В CP2 основной фокус был сделан на NLP-подходе.  
Для обучения использовались текстовые колонки:

- `title`;
- `company_profile`;
- `description`;
- `requirements`;
- `benefits`.

Текст был объединён в один признак `text`, очищен от HTML-тегов, ссылок и служебных символов, а затем преобразован в TF-IDF-признаки.


## Модели

### 1. Baseline

На этапе CP1 была построена baseline-модель:

- TF-IDF + Logistic Regression.

Также была протестирована дополнительная модель:

- TF-IDF + RandomForest.

### 2. CP2: NLP-эксперименты

На этапе CP2 были протестированы модели:

- Logistic Regression;
- Linear SVC;
- Multinomial Naive Bayes;
- Complement Naive Bayes;
- SGDClassifier.

Также был выполнен подбор гиперпараметров для:

- Linear SVC;
- Logistic Regression.

Дополнительно были проведены NLP-эксперименты:

- TF-IDF с удалением английских stop words;
- TF-IDF на character n-grams.

### 3. Финальная модель

Финальной моделью выбрана:

**TF-IDF + Linear SVC**

Параметры финальной модели:

- `TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)`;
- `LinearSVC(C=1.0, class_weight="balanced", random_state=42)`.

Финальная модель сохранена в:

```text
models/final_tfidf_linear_svc.joblib

## Результаты СР1
| Модель | Precision (fraud) | Recall (fraud) | F1-score | ROC-AUC |
|--------|------------------|----------------|----------|---------|
| Logistic Regression | 0.59 | 0.89 | 0.71 | 0.983 |
| RandomForest | 1.00 | 0.61 | 0.76 | 0.982 |


## CP2
| Модель                       | Accuracy | Precision (fraud) | Recall (fraud) | F1-score (fraud) | ROC-AUC |
| ---------------------------- | -------: | ----------------: | -------------: | ---------------: | ------: |
| Linear SVC tuned             |    0.993 |             0.951 |          0.896 |            0.923 |   0.992 |
| Logistic Regression tuned    |    0.990 |             0.893 |          0.913 |            0.903 |   0.992 |
| Linear SVC                   |    0.990 |             0.932 |          0.867 |            0.898 |   0.992 |
| Logistic Regression          |    0.983 |             0.768 |          0.919 |            0.837 |   0.990 |
| SGDClassifier                |    0.978 |             0.708 |          0.925 |            0.802 |   0.990 |
| Complement NB                |    0.964 |             0.680 |          0.491 |            0.570 |   0.944 |
| Multinomial NB               |    0.962 |             0.828 |          0.277 |            0.416 |   0.944 |
| Linear SVC tuned + stopwords |    0.992 |             0.950 |          0.873 |            0.910 |   0.992 |
| Linear SVC char ngrams       |    0.986 |             0.869 |          0.844 |            0.856 |   0.987 |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
