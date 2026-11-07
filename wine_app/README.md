# HW3 · Оценка качества вина

## Локальный запуск

Используйте Python 3.12. Из корня распакованного архива выполните команды в окружении этого проекта.

```bash
python -m pip install -r wine_app/requirements.txt
python -m streamlit run wine_app/app.py
```

По умолчанию приложение открывается на `http://localhost:8501`. `app.py` читает `model.joblib`, `config.json` и `test.csv` из своей папки. Пайплайн уже обучен. Для загрузки CSV используйте заголовки, указанные во вкладке «Пакетное предсказание», и кодировку UTF-8. Поле `recommended` содержит 1 для рекомендации и 0 для пропуска.

## Повторный запуск ноутбука

В том же окружении установите зависимости для вычислительных ячеек и Jupyter.

```bash
python -m pip install imbalanced-learn==0.14.0 seaborn==0.13.2 notebook ipykernel==6.30.1
python -m notebook HW3_MMDA.ipynb
```

Откройте ноутбук из корня архива и выполните все ячейки сверху вниз. Источник данных `Data/wine_quality.csv`. В конце ноутбук записывает модель, порог и тестовую выборку в `wine_app/`.

## Кэширование

`st.cache_resource` хранит один экземпляр модели для повторных запусков скрипта и разных сессий. При прямом вызове `joblib.load` в основном коде модель каждый раз заново читалась бы с диска и десериализовалась, даже при простом движении ползунка.

`st.cache_data` хранит таблицу и массив тестовых вероятностей как данные и возвращает их копии. `_model` исключён из хеширования, а входная таблица `X` входит в ключ кэша. Слайдер меняет сравнение готовых оценок с порогом. После замены файлов перезапустите сервер, чтобы сбросить кэши.

## Streamlit Community Cloud

1. Добавьте папку `wine_app/` целиком в GitHub-репозиторий, к которому подключён ваш аккаунт Streamlit Community Cloud.
2. Откройте [Community Cloud](https://share.streamlit.io/) и нажмите **Create app**.
3. Выберите репозиторий и ветку. Укажите `wine_app/app.py` как **Main file path**. Файл `requirements.txt` находится рядом с `app.py`.
4. В **Advanced settings** выберите Python 3.12. Нажмите **Deploy**. Откройте опубликованную страницу и приложите её ссылку к сдаче.

## Данные

Wine Quality (Cortez, Cerdeira, Almeida, Matos и Reis, 2009), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Использована [версия курса](https://github.com/paiflare/ml_hse_2026/tree/master/homework/hw_3) с `is_red` и бинарной целью для оценок от 7. Тестовая выборка получена после удаления точных дубликатов и стратифицированного разбиения.
