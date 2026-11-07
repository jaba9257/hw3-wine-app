import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Цвет вина',
}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def single_wine(model, X, threshold):
    values = {}
    cols = st.columns(3)
    for index, feature in enumerate(X.columns):
        with cols[index % len(cols)]:
            if feature == 'is_red':
                values[feature] = st.selectbox(
                    FEATURE_LABELS[feature], [0, 1], index=int(X[feature].median() >= 0.5),
                    format_func=lambda value: 'Красное' if value else 'Белое',
                )
            else:
                series = X[feature]
                values[feature] = st.number_input(
                    FEATURE_LABELS[feature], min_value=float(series.min()),
                    max_value=float(series.max()), value=float(series.median()),
                    step=0.00001 if feature == 'density' else 0.001,
                    format='%.5f',
                )
    sample = pd.DataFrame([values], columns=X.columns)
    score = float(model.predict_proba(sample)[0, 1])
    st.metric('Вероятность класса «хорошее вино»', f'{score:.2%}')
    if score >= threshold:
        st.success('Решение модели — добавить в подборку.')
    else:
        st.info('Решение модели — пропустить.')


def model_metrics(y, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    precision = metrics.precision_score(y, predicted, zero_division=0)
    recall = metrics.recall_score(y, predicted, zero_division=0)
    f1 = metrics.f1_score(y, predicted, zero_division=0)
    matrix = metrics.confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    fpr = fp / (tn + fp)

    for col, name, value in zip(st.columns(3), ('Precision', 'Recall', 'F1'), (precision, recall, f1)):
        col.metric(name, f'{value:.3f}')
    st.write(
        'При повышении порога подборка может сократиться, а recall снизиться. '
        'При снижении порога подборка расширяется, и среди добавленных вин могут оказаться посредственные.'
    )

    fig, ax = plt.subplots(figsize=(4.6, 3.8), layout='constrained')
    metrics.ConfusionMatrixDisplay(matrix, display_labels=['Класс 0', 'Класс 1']).plot(
        ax=ax, cmap='Blues', colorbar=False,
    )
    ax.set(title='Матрица ошибок', xlabel='Предсказанный класс', ylabel='Истинный класс')
    st.pyplot(fig, width=450)
    plt.close(fig)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    metrics.RocCurveDisplay.from_predictions(y, scores, name='Модель', ax=ax_roc)
    ax_roc.plot([0, 1], [0, 1], '--', color='gray', label='Случайный уровень')
    ax_roc.scatter(fpr, recall, color='#b83227', zorder=5, label='Текущий порог')
    ax_roc.set(title='ROC-кривая', xlabel='Доля FP среди класса 0', ylabel='Recall')
    ax_roc.legend(loc='lower right')

    metrics.PrecisionRecallDisplay.from_predictions(y, scores, name='Модель', ax=ax_pr)
    ax_pr.axhline(y.mean(), linestyle='--', color='gray', label='Доля класса 1')
    # При отсутствии рекомендаций показываем служебный конец PR-кривой (0, 1).
    pr_position = precision if predicted.any() else 1.0
    ax_pr.scatter(recall, pr_position, color='#b83227', zorder=5, label='Текущий порог')
    ax_pr.set(title='PR-кривая', xlabel='Recall', ylabel='Precision')
    ax_pr.legend(loc='upper right')
    st.pyplot(fig)
    plt.close(fig)
    if not predicted.any():
        st.caption('Подборка пуста. Precision условно равен 0 в карточке, на PR-кривой показана её конечная точка.')


def predict_csv(upload, model, features, threshold):
    try:
        frame = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise ValueError('Не удалось прочитать CSV. Нужен файл в UTF-8 с разделителем-запятой.') from error

    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError('В CSV отсутствуют столбцы: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В CSV есть заголовки, но нет ни одной строки с вином.')

    result = frame[features].apply(pd.to_numeric, errors='coerce')
    invalid = result.columns[~np.isfinite(result.to_numpy(dtype=float)).all(axis=0)].tolist()
    if invalid:
        raise ValueError('В этих столбцах есть пустые или нечисловые значения: ' + ', '.join(invalid))
    if not result['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red допустимы только 0 (белое) и 1 (красное).')

    scores = model.predict_proba(result)[:, 1]
    result['probability_good'] = scores
    result['recommended'] = (scores >= threshold).astype(int)
    return result


def batch_predictions(model, features, threshold):
    st.caption('CSV с заголовками, разделитель запятая. Столбец target не требуется.')
    st.code(','.join(features), language='text')
    upload = st.file_uploader('Загрузить CSV с признаками вин', type='csv')
    if upload is None:
        return
    try:
        result = predict_csv(upload, model, features, threshold)
    except ValueError as error:
        st.error(str(error))
        return
    st.dataframe(result, hide_index=True)
    st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8-sig'),
                       'wine_predictions.csv', mime='text/csv')


def model_coefficients(model, features):
    coefficients = pd.Series(model.named_steps['logisticregression'].coef_[0], index=features).sort_values()
    names = [FEATURE_LABELS[feature] for feature in coefficients.index]
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.barh(names, coefficients, color=np.where(coefficients >= 0, '#216e68', '#a14c40'))
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент при стандартизированном признаке')
    st.pyplot(fig)
    plt.close(fig)

    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    st.write('Сильнее всего повышают оценку: ' + ', '.join(FEATURE_LABELS[name] for name in positive.index) + '.')
    st.write('Сильнее всего снижают оценку: ' + ', '.join(FEATURE_LABELS[name] for name in negative.index) + '.')
    st.caption('Коэффициенты описывают вклад в логарифм шансов при фиксированных остальных признаках.')


def main():
    st.set_page_config(page_title='Оценка качества вина', layout='wide')
    st.title('Оценка качества вина')
    model = load_model()
    test = load_test_data()
    features = list(model.feature_names_in_)
    X = test[features]
    default_threshold = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))['threshold']
    threshold = st.sidebar.slider(
        'Порог рекомендации', 0.0, 1.0, value=default_threshold, step=0.000001, format='%.6f',
    )
    st.sidebar.caption('Вино попадает в подборку, если оценка модели не ниже порога.')

    single, quality, batch, coefficients = st.tabs([
        'Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель',
    ])
    with single:
        single_wine(model, X, threshold)
    with quality:
        model_metrics(test['target'], test_proba(model, X), threshold)
    with batch:
        batch_predictions(model, features, threshold)
    with coefficients:
        model_coefficients(model, features)


if __name__ == '__main__':
    main()
