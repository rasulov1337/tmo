import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Открываем CSV-файл
data = pd.read_csv("software.csv")  # Убедитесь, что имя файла совпадает
print(data.head())
print(data.info())
print(f"Количество строк до удаления дубликатов: {data.shape[0]}")

# Проверка на наличие дубликатов
print(f"Количество дубликатов: {data.duplicated().sum()}")

# Удаление дубликатов
data = data.drop_duplicates()

# Проверка после удаления
print(f"Количество строк после удаления дубликатов: {data.shape[0]}")

# Результаты агрегирования
aggregated_results = {}

# 1. Количество вакансий по компаниям
vacancies_by_company = data['Company'].value_counts().head(10)
aggregated_results['Количество вакансий по компаниям (топ-10)'] = vacancies_by_company

# 2. Распределение рейтингов компаний
avg_company_score = data['Company Score'].dropna()
aggregated_results['Распределение рейтингов компаний'] = avg_company_score

# 3. Популярные должности
popular_job_titles = data['Job Title'].value_counts().head(10)
aggregated_results['Топ-10 популярных должностей'] = popular_job_titles

# 4. Количество вакансий по местоположению
vacancies_by_location = data['Location'].value_counts().head(10)
aggregated_results['Количество вакансий по местоположению (топ-10)'] = vacancies_by_location

# 5. Распределение зарплат
data['Salary'] = pd.to_numeric(data['Salary'].str.replace('[^0-9.]', '', regex=True), errors='coerce')
aggregated_results['Распределение зарплат'] = data['Salary']

# Вывод результатов на экран и построение графиков
for title, result in aggregated_results.items():
    print(f"\n{title}:")
    print(result)

    # Визуализация результатов
    if title == 'Количество вакансий по компаниям (топ-10)':  # Круговая диаграмма
        result.plot(kind='pie', autopct='%1.1f%%', startangle=140, ylabel='', title=title, colormap="viridis")
    elif title == 'Распределение рейтингов компаний':  # Гистограмма
        plt.figure(figsize=(8, 6))
        sns.histplot(result, kde=True, bins=15, color="skyblue")
        plt.title(title)
        plt.xlabel("Рейтинг компании")
        plt.ylabel("Частота")
    elif title == 'Топ-10 популярных должностей':  # Горизонтальная столбчатая диаграмма
        plt.figure(figsize=(10, 6))
        sns.barplot(x=result.values, y=result.index, palette="viridis")
        plt.title(title)
        plt.xlabel("Количество вакансий")
        plt.ylabel("Должность")
    elif title == 'Количество вакансий по местоположению (топ-10)':  # Кольцевая диаграмма
        plt.figure(figsize=(8, 6))
        result.plot(kind='pie', autopct='%1.1f%%', startangle=140, ylabel='', title=title, colormap="coolwarm")
    elif title == 'Распределение зарплат':  # Ящик с усами
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=result, color="orange")
        plt.title(title)
        plt.xlabel("Зарплата")
    plt.show()

from scipy.stats import spearmanr, kruskal

# Проверяем данные
print(data.info())

# --- Гипотеза 1: Зависимость рейтинга компании от зарплаты ---
filtered_data = data[['Company Score', 'Salary']].dropna()
correlation, p_value = spearmanr(filtered_data['Company Score'], filtered_data['Salary'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Salary', y='Company Score', data=filtered_data, alpha=0.6)
plt.title('Зависимость рейтинга компании от зарплаты')
plt.xlabel('Зарплата')
plt.ylabel('Рейтинг компании')
plt.show()

if p_value < 0.05:
    print(f"Гипотеза 1 подтверждена: корреляция = {correlation:.2f}, p-value = {p_value:.4f}")
else:
    print(f"Гипотеза 1 не подтверждена: корреляция = {correlation:.2f}, p-value = {p_value:.4f}")

# --- Гипотеза 2: Влияние местоположения на среднюю зарплату ---
data_location_salary = data[['Location', 'Salary']].dropna()
grouped_salary = data_location_salary.groupby('Location')['Salary']
average_salaries = grouped_salary.mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Location', y='Salary', data=data_location_salary, palette='coolwarm')
plt.title('Распределение зарплат по местоположению')
plt.xticks(rotation=45)
plt.show()

stat, p_value = kruskal(*[group['Salary'].values for _, group in data_location_salary.groupby('Location')])

if p_value < 0.05:
    print("Гипотеза 2 подтверждена: местоположение влияет на зарплату")
else:
    print("Гипотеза 2 не подтверждена: местоположение не влияет на зарплату")

# --- Гипотеза 3: Влияние должности на рейтинг компании ---
data_job_score = data[['Job Title', 'Company Score']].dropna()
average_scores = data_job_score.groupby('Job Title')['Company Score'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=average_scores.values, y=average_scores.index, palette='viridis')
plt.title('Средний рейтинг компании по должностям')
plt.xlabel('Рейтинг компании')
plt.ylabel('Должность')
plt.show()

# Проводить статистические тесты сложно из-за большого количества категорий, но
# визуализация показывает различия в среднем рейтинге для топ-10 должностей.
print("Гипотеза 3: Анализ завершен, различия наблюдаются визуально.")


# --- Гипотеза 6: Зависимость зарплаты от должности ---
# Выбираем данные с указанной зарплатой и должностями
job_salary_data = data[['Job Title', 'Salary']].dropna()

# Группируем данные по должностям
salary_by_job = job_salary_data.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(10)

# Визуализация топ-10 должностей по средней зарплате
plt.figure(figsize=(10, 6))
sns.barplot(x=salary_by_job.values, y=salary_by_job.index, palette='coolwarm')
plt.title('Средняя зарплата для топ-10 должностей')
plt.xlabel('Средняя зарплата')
plt.ylabel('Должность')
plt.show()


# --- Гипотеза 7: Указывают ли компании зарплату чаще в определенных локациях? ---
# Вычисляем долю вакансий с указанной зарплатой по локациям
location_salary_data = data[['Location', 'Salary']].copy()
location_salary_data['Salary Specified'] = location_salary_data['Salary'].notnull()

# Группируем данные по местоположению
salary_specified_ratio = location_salary_data.groupby('Location')['Salary Specified'].mean().sort_values(ascending=False).head(10)

# Визуализация доли вакансий с указанной зарплатой
plt.figure(figsize=(12, 8))
sns.barplot(x=salary_specified_ratio.values, y=salary_specified_ratio.index, palette='viridis')
plt.title('Доля вакансий с указанной зарплатой (топ-10 местоположений)')
plt.xlabel('Доля вакансий с зарплатой')
plt.ylabel('Местоположение')
plt.show()

# Проверяем гипотезу
if salary_specified_ratio.var() > 0:
    print("Гипотеза 7 подтверждена: доля указанных зарплат варьируется в зависимости от местоположения.")
else:
    print("Гипотеза 7 не подтверждена: доля указанных зарплат не зависит от местоположения.")


# Используем только числовые колонки для корреляции
numeric_data = data[['Company Score', 'Salary']]

# Корреляция Пирсона между числовыми столбцами
pearson_corr = numeric_data.corr(method='pearson')

# Корреляция Спирмена между числовыми столбцами
spearman_corr = numeric_data.corr(method='spearman')

# Печать матриц корреляции
print("Корреляция Пирсона:")
print(pearson_corr)
print("\nКорреляция Спирмена:")
print(spearman_corr)

# Визуализация с помощью тепловой карты
plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Тепловая карта корреляции Пирсона")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Тепловая карта корреляции Спирмена")
plt.show()