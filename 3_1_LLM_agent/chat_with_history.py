import requests
import json
from decouple import config

# НастройGhки подключения
API_KEY = config('OPENROUTER_API_KEY')
URL = "https://openrouter.ai/api/v1/chat/completions"
# Укажите нужную модель (например, актуальную gpt-4o, mistral или gpt-5, если доступна)
MODEL_NAME = "openai/gpt-5.4-nano" 

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Инициализируем историю сообщений системным промптом
messages = [
    {"role": "system", "content": "Ты — полезный и вежливый ИИ-ассистент."}
]

print("--- Чат запущен! Введите 'exit' для выхода. ---")

while True:
    # 2. Получаем ввод от пользователя
    user_input = input("\nВы: ")
    
    # Проверка на выход из программы
    if user_input.strip().lower() == 'exit':
        print("Чат завершен.")
        break
        
    if not user_input.strip():
        continue

    # 3. Добавляем сообщение пользователя в историю
    messages.append({"role": "user", "content": user_input})

    # Подготовка данных для отправки (вся история целиком)
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7
    }

    try:
        # 4. Отправляем запрос к API
        response = requests.post(URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Извлекаем текст ответа нейросети
            assistant_reply = response_data['choices'][0]['message']['content']
            
            # Выводим ответ в консоль
            print(f"Бот: {assistant_reply}")
            
            # 5. Самый важный шаг: сохраняем ответ модели в историю для следующего шага
            messages.append({"role": "assistant", "content": assistant_reply})
            
        else:
            print(f"\n[Ошибка API {response.status_code}]: {response.text}")
            # Удаляем последнее сообщение пользователя, так как модель на него не ответила
            messages.pop()
            
    except Exception as e:
        print(f"\n[Ошибка сети]: {e}")
        messages.pop()
