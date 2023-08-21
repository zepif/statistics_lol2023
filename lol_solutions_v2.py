"""lol_solutions_v2.ipynb

Original file is located at
    https://colab.research.google.com/drive/1A6SA3a9lIFgbYOrJ4dpl4hykYWB-_193

#**reading data**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.manifold import TSNE

if not os.path.exists('templates/png_files'):
    os.makedirs('templates/png_files')

if not os.path.exists('templates/csv_files'):
    os.makedirs('templates/csv_files')

file_path = "LOL2023_solutions_last.csv"

data = pd.read_csv(file_path, sep=';', header=0, error_bad_lines = False)

print(data.head(10))

"""#**preprocessing data**"""
def preprocessing(data):
    data = data.drop(['filename'], axis=1, inplace=False)
    data = data.drop(['checksum'], axis=1, inplace=False)
    data = data.drop(['is_passed'], axis=1, inplace=False)
    data = data.drop(['module_val'], axis=1, inplace=False)
    data = data.drop(['check_type'], axis=1, inplace=False)
    data = data.drop(['test_score'], axis=1, inplace=False)

    data['problem_level'] = data['problem_id'] % 10
    data['problem_level'] = data['problem_level'].apply(lambda x: 'L1' if (x <= 4 and x != 0) else ('L2' if (x <= 7 and x != 0) else 'L3'))

    json_file_path = 'class.json'

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    json_data = {int(key): value for key, value in json_data.items()}

    data['class'] = data['user_id'].map(json_data)

    return data

#data.to_csv('clear_df.csv', index=False)

data = preprocessing(data)

print(data.head(10))

"""#1ая задача"""

data['problem_level'] = data['problem_id'] % 10
data['problem_level'] = data['problem_level'].apply(lambda x: 'L1' if (x <= 4 and x != 0) else ('L2' if (x <= 7 and x != 0) else 'L3'))

result = data.groupby(['problem_id', 'user_id', 'problem_level'])['score'].max().reset_index()

result = result.sort_values(by=['user_id', 'problem_id'])
print(result)
sum_by_level = result.groupby(['user_id', 'problem_level'])['score'].sum().unstack(fill_value=0)

sum_by_level.columns = ['L1', 'L2', 'L3']

sum_by_level.reset_index(inplace=True)
print(sum_by_level)
#sum_by_level.to_csv('templates/csv_files/solved_tasks.csv', index=False, encoding='utf-8')

"""#2ая задача"""

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

language = {
    4 : "pascal",
    11 : "python2",
    12 : "python3",
    16 : "GO",
    17 : "Java 8",
    18 : "C",
    23 : "rust",
    24 : "scala",
    26 : "kotlin",
    28 : "python-ml",
    35 : ".NET C#",
    37 : "C++",
    39 : "delphi"
}

data['lang'] = data['lang_id'].map(language)

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='lang')
plt.title("Кількість рішень за мовами програмування")
plt.xlabel("Мова програмування")
plt.ylabel("Кількість рішень")
plt.xticks(rotation=45)
#plt.show()
plt.tight_layout()
plt.savefig('templates/png_files/num_solutions.png', bbox_inches='tight')
plt.close()

data['posted_time'] = pd.to_datetime(data['posted_time'], unit='s')
data['day'] = data['posted_time'].dt.date

date = pd.to_datetime('2023-08-08').date()
data['day'] = data['day'].apply(lambda x : x if (x >= date) else None)

data = data[data['day'].notnull()].sort_values(by='day')

plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='day')
plt.title("Кількість розміщених рішень за днями")
plt.xlabel("Дата")
plt.ylabel("Кількість рішень")
plt.xticks(rotation=45)
#plt.show()
plt.tight_layout() 
plt.savefig('templates/png_files/num_attempts.png', bbox_inches='tight')
plt.close()

lang_counts = data['lang'].value_counts()

significant_langs = lang_counts[lang_counts >= lang_counts.sum() * 0.01]

plt.figure(figsize=(8, 8))
plt.pie(significant_langs, labels=significant_langs.index, autopct='%1.1f%%', startangle=140)
plt.title("Розподіл рішень за мовами програмування")
#plt.show()
plt.savefig('templates/png_files/count_by_lang.png')
plt.close()

day_counts = data['day'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(day_counts, labels=day_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Розподіл розміщених рішень за днями")
#plt.show()
plt.savefig('templates/png_files/daily_distribution.png')
plt.close()


verdicts = {
    2 : 'CE',
    4 : 'WA',
    8 : 'PE',
    16 : 'TL',
    32 : 'ML',
    64 : 'RE',
    128 : 'FF',
    256 : 'ZR'
}

def get_verdict(test_result):
    result = []
    if test_result == 0:
        result.append('OK')
        return result

    for key, value in verdicts.items():
        if test_result & key:
            result.append(value)
    return result

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

data['verdict'] = data['test_result'].apply(get_verdict)

verdict_counts = data['verdict'].value_counts()

plt.figure(figsize=(10, 6))
verdict_counts.plot(kind='bar')
plt.title('Кількість рішень за вердиктами')
plt.xlabel('Вердикт')
plt.ylabel('Кількість рішень')
#plt.show()
plt.tight_layout() 
plt.savefig('templates/png_files/lang_distribution.png', bbox_inches='tight')
plt.close()

error_counts = {}
for verdict_list in data['verdict']:
    for verdict in verdict_list:
        error_counts[verdict] = error_counts.get(verdict, 0) + 1

filtered_error_counts = {verdict: count for verdict, count in error_counts.items() if count >= 0.01 * len(data)}

plt.figure(figsize=(8, 8))
plt.pie(filtered_error_counts.values(), labels=filtered_error_counts.keys(), autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(filtered_error_counts.values()) / 100), startangle=140)
plt.title('Розподіл помилок')
#plt.show()
plt.savefig('templates/png_files/error_distribution.png')
plt.close()

total_count = len(data)
print(f"Загальна кількість рішень: {total_count}")

"""#3&4 задача"""

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

data['problem_level'] = data['problem_id'] % 10
data['problem_level'] = data['problem_level'].apply(lambda x: 'L1' if (x <= 4 and x != 0) else ('L2' if (x <= 7 and x != 0) else 'L3'))

levels = ['L1', 'L2', 'L3']

analysis_results = {}

for level in levels:
    level_data = data[data['problem_level'] == level]
    total_solutions = len(level_data)
    avg_score = level_data['score'].mean()
    success_rate = (level_data['score'] == 100).sum() / total_solutions * 100
    error_rate = (level_data['score'] < 15).sum() / total_solutions * 100

    analysis_results[level] = {
        'total_solutions' : total_solutions,
        'avg_score' : avg_score,
        'success_rate' : success_rate,
        'error_rate' : error_rate
    }

for level, results in analysis_results.items():
    plt.figure(figsize=(10,6))
    plt.title(f"Аналіз для рівня завдань {level}")
    plt.bar(['Середній бал','Успішні {%}', 'Помилкові {%}'],
            [results['avg_score'], results['success_rate'], results['error_rate']])
    plt.ylim(0, 110)
    plt.ylabel("Значення")
    plt.ylabel("Метрика")
    #plt.show()
    plt.savefig(f'templates/png_files/analysis_{level}.png')
    plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(data['checked_time'], data['score'], alpha=0.5)
plt.xlabel('Час перевірки')
plt.ylabel('Бали')
plt.title('Час перевірки vs. Бали')
plt.tight_layout()
#plt.show()
plt.savefig('templates/png_files/test_time_vs_score.png')
plt.close()



plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.figure(figsize=(10, 6))
problem_counts = data['problem_id'].value_counts()
sns.barplot(x=problem_counts.index, y=problem_counts.values)
plt.xlabel('Ідентифікатор завдання')
plt.ylabel('Кількість рішень')
plt.title('Кількість рішень за завданнями')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig('templates/png_files/num_tasks_solutions.png')
plt.close()

compile_errors = data[data['compile_error'].notnull()]
plt.figure(figsize=(10, 6))
compile_error_counts = compile_errors['problem_id'].value_counts()
sns.barplot(x=compile_error_counts.index, y=compile_error_counts.values)
plt.xlabel('Ідентифікатор завдання')
plt.ylabel('Кількість помилок компіляції')
plt.title('Кількість неправильних рішень за завданнями')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig('templates/png_files/num_errors.png')
plt.close()

data = data[data['score'] == 100]
problem_counts = data[data['problem_level'].isin(['L1', 'L2', 'L3'])].groupby(['problem_id', 'problem_level', 'score'])['user_id'].nunique()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data[data['problem_level'].isin(['L1', 'L2', 'L3'])], x='problem_id', hue='problem_level')
ax.set_title('Кількість людей, які розв\'язали задачі L1, L2 i L3')
ax.set_xlabel('Номер завдання')
ax.set_ylabel('Кількість людей')
ax.legend(title='Рівень завдання')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.show()
plt.savefig('templates/png_files/num_people_solved_l123.png')
plt.close()

"""#5ая задач

"""

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

'''nums = [3123, 3124, 3125, 3126, 3127, 3129, 3130, 3131, 3132, 3133, 3135, 3136,
        3137, 3138, 3139, 3141, 3142, 3143, 3144, 3145, 3147, 3148, 3149, 3150,
        3151]'''
nums = []
count = 0
for i in range(3123, 4000):
    if count < 5:
        nums.append(i)
        count += 1
    else:
        count = 0

def transform_number(num):
    for i in range(100):
        if (num == nums[i]):
            t = i
            break
    return f"{t%5+6:02d}#{t//5+1}"

data['contest'] = data['contest_id'].apply(transform_number)

data['posted_time'] = pd.to_datetime(data['posted_time'], unit='s')

date = pd.to_datetime('2023-08-08').date()
data['posted_time'] = data['posted_time'].apply(lambda x : x if (x >= date) else None)

data = data[data['score'] == 100].drop_duplicates(subset=['user_id', 'problem_id'])

max_score_users = data[data['score'] == 100]

solved_tasks_per_day_per_user = max_score_users.groupby([pd.Grouper(key='posted_time', freq='D'), 'user_id']).agg({'problem_id': 'count', 'score': 'sum'})

max_solved_tasks_per_day = solved_tasks_per_day_per_user.groupby('posted_time')['problem_id'].transform('max')

top_users_per_day = solved_tasks_per_day_per_user[solved_tasks_per_day_per_user['problem_id'] == max_solved_tasks_per_day]

top_users_per_day_df = top_users_per_day.reset_index()
top_users_per_day_df = top_users_per_day_df.drop(columns = ['score'], axis = 1)
column_rename_mapping = {'posted_time': 'День', 'user_id': 'ID користувача', 'problem_id' : 'ID задачі'}
top_users_per_day_df.rename(columns=column_rename_mapping, inplace=True)
top_users_per_day_df.to_csv('templates/csv_files/топ користувачів за день.csv', index=False, encoding='utf-8')

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

data['posted_time'] = pd.to_datetime(data['posted_time'], unit='s')
data['checked_time'] = pd.to_datetime(data['checked_time'], unit='s')

data.set_index('posted_time', inplace=True)

hourly_resampled = data.resample('H').size()
daily_resampled = data.resample('D').size()

hourly_ok_resampled = data[data['test_result'] == 0].resample('H').size()
daily_ok_resampled = data[data['test_result'] == 0].resample('D').size()

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
hourly_resampled.plot(label='Кількість рішень')
hourly_ok_resampled.plot(label='Кількість ОК')
plt.xlabel('Час (по годинах)')
plt.ylabel('Кількість')
plt.title('Кількість рішень і ОК по годинах')
plt.legend()

plt.subplot(2, 1, 2)
daily_resampled.plot(label='Кількість рішень')
daily_ok_resampled.plot(label='Кількість ОК')
plt.xlabel('Дата (по днях)')
plt.ylabel('Кількість')
plt.title('Кількість рішень і ОК за днями')
plt.legend()

plt.tight_layout()
#plt.show()
plt.savefig('templates/png_files/num_solved_h_d.png')
plt.close()


"""#таблицы

"""
#1ая таблица
data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

filtered_df = data[data['class'] != 'root']

class_order = ["5", "6", "7", "8", "9", "10+"]


result_df = filtered_df.groupby('class').agg({
    'user_id': 'nunique',
    'solution_id': 'count',
    'score': [lambda x: (x == 100).sum(), 'sum']
}).reindex(class_order, fill_value=0).reset_index()

result_df.columns = ['клас', 'кількість користувачів', 'кількість спроб', 'успішні відправлення', 'сума балів']

print(result_df)

result_df.to_csv('templates/csv_files/результати відправок за класами.csv', index=False, encoding='utf-8')


#2ая таблица
data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

grouped = data.groupby("problem_level").agg(
    num_problems=("problem_id", "nunique"),
    num_participants=("user_id", "nunique"),
    num_solutions=("solution_id", "count"),
    num_ok_solutions=("score", lambda x: (x == 100.0).sum())
).reset_index()

column_rename_mapping = {'problem_level': 'рівень задачи', 'num_problems': 'кількість задач', 'num_participants' : 'кількість учасників',
                         'num_solutions' : 'кількість розв\'язків', 'num_ok_solutions' : 'кількість правильних розв\'язків'}
grouped.rename(columns=column_rename_mapping, inplace=True)

print(grouped)
grouped.to_csv('templates/csv_files/результати відправок за групами.csv', index=False, encoding='utf-8')


#3ья таблица
data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

#data = data[data['score'] == 100]

difficulty_counts = {
    'L1': [0, 0, 0, 0, 0],  # >= 30%, >= 50%, >= 60%, >= 74%, >= 90%
    'L2': [0, 0, 0, 0, 0],
    'L3': [0, 0, 0, 0, 0]
}
unique_users_l1 = set()
unique_users_l2 = set()
unique_users_l3 = set()
#user_levels = {}
for index, row in data.iterrows():
    user_id = row['user_id']
    difficulty = row['problem_level']
    percent_solved = row['score']

    '''if user_id in user_levels and user_levels[user_id] == difficulty:
        continue

    user_levels[user_id] = difficulty'''

    if difficulty == 'L1' and user_id not in unique_users_l1:
        unique_users_l1.add(user_id)
        if percent_solved >= 30:
            difficulty_counts['L1'][0] += 1
        if percent_solved >= 50:
            difficulty_counts['L1'][1] += 1
        if percent_solved >= 60:
            difficulty_counts['L1'][2] += 1
        if percent_solved >= 74:
            difficulty_counts['L1'][3] += 1
        if percent_solved >= 90:
            difficulty_counts['L1'][4] += 1
    elif difficulty == 'L2' and user_id not in unique_users_l2:
        unique_users_l2.add(user_id)
        if percent_solved >= 30:
            difficulty_counts['L2'][0] += 1
        if percent_solved >= 50:
            difficulty_counts['L2'][1] += 1
        if percent_solved >= 60:
            difficulty_counts['L2'][2] += 1
        if percent_solved >= 74:
            difficulty_counts['L2'][3] += 1
        if percent_solved >= 90:
            difficulty_counts['L2'][4] += 1
    elif difficulty == 'L3' and user_id not in unique_users_l3:
        unique_users_l3.add(user_id)
        if percent_solved >= 30:
            difficulty_counts['L3'][0] += 1
        if percent_solved >= 50:
            difficulty_counts['L3'][1] += 1
        if percent_solved >= 60:
            difficulty_counts['L3'][2] += 1
        if percent_solved >= 74:
            difficulty_counts['L3'][3] += 1
        if percent_solved >= 90:
            difficulty_counts['L3'][4] += 1



difficulty_counts = pd.DataFrame.from_dict(difficulty_counts, orient='index', columns=['набрано балів >= 30%', 'набрано балів >= 50%', 
                                                                                       'набрано балів >= 60%', 'набрано балів >= 74%', 
                                                                                       'набрано балів >= 90%'])

difficulty_counts['Рівень'] = ["L1", "L2", "L3"]
difficulty_counts= difficulty_counts[['Рівень'] + [col for col in difficulty_counts if col != 'Рівень']]

print(difficulty_counts)
difficulty_counts.to_csv('templates/csv_files/результати за групами.csv', index=False, encoding='utf-8')

#4ая табличка 
data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

cp = data
solved_problems = data[data['score'] == 100].drop_duplicates(subset=['user_id', 'problem_id'])

grouped_data = data.groupby('problem_id')

attempts_count = {}
solved_count = {}
total_scores = {}
problem_levels = {}

for problem_id, group in grouped_data:
    attempts_count[problem_id] = len(group)
    total_scores[problem_id] = group['score'].sum()
    problem_levels[problem_id] = group['problem_level'].iloc[0]

grouped_solved = solved_problems.groupby('problem_id')

for problem_id, group in grouped_solved:
    solved_count[problem_id] = len(group)

attempts_df = pd.DataFrame(list(attempts_count.items()), columns=['Problem ID', 'Спроби'])
solved_df = pd.DataFrame(list(solved_count.items()), columns=['Problem ID', 'Вирішено на 100'])
scores_df = pd.DataFrame(list(total_scores.items()), columns=['Problem ID', 'Сумарні бали'])
levels_df = pd.DataFrame(list(problem_levels.items()), columns=['Problem ID', 'Рівень'])

result_df = attempts_df.merge(solved_df, on='Problem ID', how='outer').fillna({'Вирішено на 100': 0}).merge(scores_df, on='Problem ID').merge(levels_df, on='Problem ID')
column_rename_mapping = {'Problem ID': 'ID завдання'}
result_df.rename(columns=column_rename_mapping, inplace=True)
#print(result_df)

easiest_L1 = result_df[result_df['Рівень'] == 'L1'].nsmallest(1, 'Вирішено на 100')
easiest_L2 = result_df[result_df['Рівень'] == 'L2'].nsmallest(1, 'Вирішено на 100')
easiest_L3 = result_df[result_df['Рівень'] == 'L3'].nsmallest(1, 'Вирішено на 100')

hardest_L1 = result_df[result_df['Рівень'] == 'L1'].nlargest(1, 'Вирішено на 100')
hardest_L2 = result_df[result_df['Рівень'] == 'L2'].nlargest(1, 'Вирішено на 100')
hardest_L3 = result_df[result_df['Рівень'] == 'L3'].nlargest(1, 'Вирішено на 100')

result_table_2 = pd.concat([hardest_L1, hardest_L2, hardest_L3])

result_table_2.reset_index(drop=True, inplace=True)

print(result_table_2[['Рівень', 'ID завдання', 'Вирішено на 100', 'Спроби', 'Сумарні бали']])

result_table_2.to_csv('templates/csv_files/найпростіші завдання.csv', index=False, encoding='utf-8')

result_table = pd.concat([easiest_L1, easiest_L2, easiest_L3])

result_table.reset_index(drop=True, inplace=True)

print(result_table[['Рівень', 'ID завдання', 'Вирішено на 100', 'Спроби', 'Сумарні бали']])

result_table.to_csv('templates/csv_files/найскладніші завдання.csv', index=False, encoding='utf-8')

#5ая табличка 
data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

user_submission_counts = data.groupby('user_id').size().reset_index(name='кількість відправлень')

top_10_users = user_submission_counts.sort_values(by='кількість відправлень', ascending=False).head(10)
column_rename_mapping = {'user_id': 'ID користовуча'}
top_10_users.rename(columns=column_rename_mapping, inplace=True)
print(top_10_users)

top_10_users.to_csv('templates/csv_files/топ учасників за кількістю відправлень.csv', index=False, encoding='utf-8')

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

user_scores = data.groupby('user_id')['score'].sum().reset_index()

user_scores_sorted = user_scores.sort_values(by='score', ascending=False)

top_10_users = user_scores_sorted.head(10)
column_rename_mapping = {'user_id': 'ID користовуча', 'score' : 'бал'}
top_10_users.rename(columns=column_rename_mapping, inplace=True)
print(top_10_users)

top_10_users.to_csv('templates/csv_files/топ учасників за набраними балами.csv', index=False, encoding='utf-8')

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

data = data[data['score'] == 100].drop_duplicates(subset=['user_id', 'problem_id'])

filtered_df = data[data['score'] == 100]

grouped_df = filtered_df.groupby(['class', 'user_id'])['score'].count().reset_index()

sorted_df = grouped_df.sort_values(by=['class', 'score'], ascending=[True, False])

top_users = {}

for class_name in sorted_df['class'].unique():
    class_data = sorted_df[sorted_df['class'] == class_name]
    top_3_users = class_data.head(3)
    top_users[class_name] = top_3_users

top_users_df = pd.concat(top_users.values())

top_users_df = top_users_df.head(10)
top_users_df = top_users_df.sort_values(by=['user_id'])
column_rename_mapping = {'class' : 'клас', 'user_id': 'ID користовуча', 'score' : 'бал'}
top_users_df.rename(columns=column_rename_mapping, inplace=True)
print(top_users_df)

top_users_df.to_csv('templates/csv_files/топ 3 у кожній паралелі.csv', index=False, encoding='utf-8')

#6ая табличка 
data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

print(data)

time_intervals = {
    "00:00 - 02:00": (0, 2),
    "02:00 - 04:00": (2, 4),
    "04:00 - 06:00": (4, 6),
    "06:00 - 08:00": (6, 8),
    "08:00 - 10:00": (8, 10),
    "10:00 - 12:00": (10, 12),
    "12:00 - 14:00": (12, 14),
    "14:00 - 16:00": (14, 16),
    "16:00 - 18:00": (16, 18),
    "18:00 - 20:00": (18, 20),
    "20:00 - 22:00": (20, 22),
    "22:00 - 00:00": (22, 0),
}

data['posted_time'] = pd.to_datetime(data['posted_time'], unit='s')
data['hour'] = (data['posted_time'].dt.hour + 1) % 24

data['time_interval'] = data['hour'].apply(
    lambda x: next(
        (interval for interval, (start, end) in time_intervals.items() if (start <= end and start <= x < end) or (start > end and (start <= x or x < end))),
        None
    )
)

score_100_df = data[data['score'] == 100.0]

result = data.groupby(['time_interval'])[['solution_id']].count().join(
    score_100_df.groupby(['time_interval'])[['solution_id']].count(),
    rsuffix='_score_100'
)

result.fillna(0, inplace=True)

result.columns = ['Submissions', 'Solutions with Score 100']

result = result.T

result['відправки'] = ["Усього:", "ОК:"]
result= result[['відправки'] + [col for col in result if col != 'відправки']]

print(result)
result.to_csv('templates/csv_files/кількість відправок за годинами.csv', index=False, encoding='utf-8')

data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

data['posted_time'] = pd.to_datetime(data['posted_time'], unit='s')

time_intervals = {
    "1ий День": pd.to_datetime("2023-08-08").date(),
    "2ий День": pd.to_datetime("2023-08-09").date(),
    "3iй День": pd.to_datetime("2023-08-10").date(),
    "4ий День": pd.to_datetime("2023-08-11").date(),
    "5ий День": pd.to_datetime("2023-08-12").date(),
    "6ий День": pd.to_datetime("2023-08-13").date(),
    "7ий День": pd.to_datetime("2023-08-14").date(),
    "8ий День": pd.to_datetime("2023-08-15").date(),
    "9ий День": pd.to_datetime("2023-08-16").date(),
    "10ий День": pd.to_datetime("2023-08-17").date(),
    "11ий День": pd.to_datetime("2023-08-18").date(),
    "12ий День": pd.to_datetime("2023-08-19").date(),
    "13ий День": pd.to_datetime("2023-08-20").date(),
    "14ий День": pd.to_datetime("2023-08-21").date(),
}

score_100_df = data[data['score'] == 100.0]

data['day'] = data['posted_time'].dt.date
score_100_df['day'] = score_100_df['posted_time'].dt.date

data['time_interval'] = data['day'].apply(lambda x: next((interval for interval, day in time_intervals.items() if day == x), None))
score_100_df['time_interval'] = score_100_df['day'].apply(lambda x: next((interval for interval, day in time_intervals.items() if day == x), None))

result = data.groupby(['time_interval'])[['solution_id']].count().join(score_100_df.groupby(['time_interval'])[['solution_id']].count(), rsuffix='_score_100')

result.fillna(0, inplace=True)

result.columns = ['Submissions', 'Solutions with Score 100']

#result['День'] = [key for key in time_intervals.keys()]

result = result.T

result['відправки'] = ["Усього:", "ОК:"]
result= result[['відправки'] + [col for col in result if col != 'відправки']]

print(result)

result.to_csv('templates/csv_files/кількість відправлень за днями.csv', index=False, encoding='utf-8')


#7ая табличка 
data = pd.read_csv("LOL2023_solutions_last.csv", sep=';', header=0, error_bad_lines = False)
data = preprocessing(data)

data['lang'] = data['lang_id'].map(language)

agg_funcs = {
    'user_id': 'nunique', 
    'solution_id': 'count',
    'score': 'sum',
}


stats_table = data.groupby('lang').agg(agg_funcs).reset_index()


perfect_score_solutions = data[data['score'] == 100].groupby('lang')['solution_id'].count().reset_index()
perfect_score_solutions.columns = ['lang', 'Perfect_Score_Solutions']
stats_table = pd.merge(stats_table, perfect_score_solutions, on='lang', how='left')


filtered_df = data[data['test_result'] == 2]
compile_errors = filtered_df.groupby('lang')['solution_id'].count().reset_index()
stats_table = pd.merge(stats_table, compile_errors, on='lang', how='left')


stats_table.columns = [
    'Мова програмування',
    'Кількість користувачів',
    'Усього рішень',
    'Усього балів',
    'Усього правильних рішень',
    'Помилки компіляції',
]

stats_table['Усього правильних рішень'] = stats_table['Усього правильних рішень'].fillna(0)

stats_table['Помилки компіляції'] = stats_table['Помилки компіляції'].fillna(0)
stats_table['Помилки компіляції'] = stats_table.apply(lambda row: f"{int(row['Помилки компіляції'])} ({(row['Помилки компіляції'] / row['Усього рішень']) * 100:.2f}%)", axis=1)


desired_column_order = [
    'Мова програмування',
    'Кількість користувачів',
    'Усього рішень',
    'Усього правильних рішень',
    'Усього балів',
    'Помилки компіляції',
]
stats_table = stats_table[desired_column_order]
stats_table['Усього правильних рішень'] = stats_table.apply(lambda row: f"{int(row['Усього правильних рішень'])} ({(row['Усього правильних рішень'] / row['Усього рішень']) * 100:.2f}%)", axis=1)

print(stats_table)

stats_table.to_csv('templates/csv_files/статистика за мовами програмування.csv', index=False, encoding='utf-8')