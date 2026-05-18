# Лабораторная работа №4
## Классификация с применением Scikit-Learn 


### Тема: 
Предсказание дефолта по кредиту с помощью машинного обучения

### Цель:
Научиться строить и оценивать модели классификации

### Задание:
Заполнить пропуски в коде, исследовать модели, подготовить отчет

### Код:
[Открыть ноутбук в Google Colab](https://colab.research.google.com/drive/1MOf-N_PUKxquqlMsXGH09AslP9H9cBwK#scrollTo=ovsDLCkvcwUp)
### Выполнение работы:
#### *Пункт №5:*
**Пошаговый алгоритм:**

```bash
# 1. Сохранить модель
joblib.dump(random_forest_model, 'model.pkl')

# 2. Установить зависимости
pip install fastapi uvicorn joblib pandas scikit-learn

# 3. Создать файл app.py (см. код ниже)

# 4. Запустить сервер
uvicorn app:app --host 0.0.0.0 --port 8000

# 5. Отправить тестовый запрос
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35, "MonthlyIncome":50000, "DebtRatio":0.3}'
```

**Пример кода app.py:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')

class ClientData(BaseModel):
    age: int
    MonthlyIncome: float
    DebtRatio: float

@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0, 1]
    prediction = 1 if prob >= 0.5 else 0
    return {
        "prediction": prediction,
        "default_probability": prob
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Ожидаемый ответ:**

```json
{
    "prediction": 0,
    "default_probability": 0.15
}
```

### Вывод:
Случайный лес является оптимальной моделью для данной задачи кредитного скоринга, обеспечивая хороший баланс между качеством (ROC-AUC=0.8396), скоростью обучения и устойчивостью к дисбалансу классов.