import json
import csv
from tqdm import tqdm
from custom_sentiment import calculate_sentiment

# === Вхідні файли ===
FILES = [
    ('positive.json', 'positive'),
    ('negative.json', 'negative')
]

# === Вихідний файл ===
OUTPUT_CSV = 'features_dataset.csv'

# === Заголовки колонок ===
HEADERS = [
    'positive', 'negative', 'neutral', 'compound',
    'emoji_score', 'num_boosters', 'has_negation',
    'num_tokens', 'label'
]

total_lines = 0
for file_path, _ in FILES:
    with open(file_path, encoding='utf-8') as f:
        total_lines += sum(1 for _ in f)

print(f"🔢 Всього твітів до обробки: {total_lines}")

# === Записуємо CSV ===
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
    writer.writeheader()

    with tqdm(total=total_lines, desc="🔄 Обробка твітів") as pbar:
        for file_path, label in FILES:
            with open(file_path, encoding='utf-8') as f:
                for idx, line in enumerate(f, start=1):
                    obj = json.loads(line)
                    text = obj.get('msg', '').strip()

                    # Пропускаємо через кастомний аналайзер
                    result = calculate_sentiment(text)

                    row = {
                        'positive': result.get('positive', 0),
                        'negative': result.get('negative', 0),
                        'neutral': result.get('neutral', 0),
                        'compound': result.get('compound', 0),
                        'emoji_score': result.get('emoji_score', 0),
                        'num_boosters': result.get('num_boosters', 0),
                        'has_negation': result.get('has_negation', 0),
                        'num_tokens': result.get('num_tokens', 0),
                        'label': label
                    }

                    writer.writerow(row)
                    pbar.update(1)

                    if idx % 1000 == 0:
                        print(f"✅ Опрацьовано {idx} рядків з файлу {file_path}")

print(f"🎉 Готово! CSV збережено: {OUTPUT_CSV}")
