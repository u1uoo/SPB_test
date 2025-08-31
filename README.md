## Требования
- Python 3.8+
### Установка
```bash
python -m pip install pandas matplotlib requests yfinance
```

## Структура
- `get_<instruments>.py` — загрузка данных через API Binance
- `plot.py` — построение графиков на данных из CSV файлов
- `metrics.py` — индикаторы: `sma`, `ema`, `macd`, `rsi`
- `main.py` — CLI для управления

## Примеры
Файл запускается из корня проекта

### Загрузка данных
```bash
python main.py fetch BTCUSDT --timeframe 1d
```
- Результат сохранится в `ohlc_BTCUSDT.csv` 

### Построение графиков
```bash
python main.py plot --symbol BTCUSDT
```
- Дополнительно можно передать периоды для индикаторов
```bash
python main.py plot --symbol BTCUSDT --sma 12,26 --ema 21,55
```

### ОДновременно загрузить данные и построить графики
```bash
python main.py fetch-plot BTCUSDT --timeframe 1d --sma 12,26 --ema 21,55
```


### Взаимодействие с графиками
- Колёсико мыши: зум по X на всех панелях
- Shift + колёсико: зум по Y на наведённой панели
- ЛКМ + перетаскивание: масштабирование по X на всех панелях
- Shift + ЛКМ + перетаскивание: масштабирование по Y только на наведённой панели
- Справа — чекбоксы для включения/выключения графиков

