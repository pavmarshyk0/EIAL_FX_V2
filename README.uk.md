[English](README.md) | **Українська** | [Polski](README.pl.md)

# EIAL-FX — дослідження інформаційно-алгоритмічних факторів валютного ринку

**Статус: `NO_EDGE_ESTABLISHED`.** Це експериментальна Python-система, а не прибуткова стратегія. Підтверджених результатів на реальному ринку поки немає; демонстраційні дані синтетичні.

Досліджуємо EUR/USD і EUR/PLN, а USD/PLN розраховуємо через валютний трикутник. Фактори: минулі ціни, ставки, доступні інституційні дані, sentiment, ліквідність та мікроструктура. Кожен зовнішній показник використовується **лише після часу його фактичної доступності**.

## Запуск

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python eialfx.py demo --out results/demo
python quote_backtest.py demo --out results/quote_demo
```

**Реальна історія ECB:**

```bash
python eialfx.py fetch-ecb --start 2016-01-01 --out data/ecb_fx.csv
python eialfx.py train --rates data/ecb_fx.csv --out results/ecb
```

**Торговельний бектест із реальними bid/ask:**

```bash
python dukascopy_to_quotes.py --input raw_eurusd.csv --pair eurusd --out data/eurusd_quotes.csv
python quote_backtest.py run --predictions results/ecb/oos_predictions.csv --rates data/ecb_fx.csv --quotes data/eurusd_quotes.csv --out results/quotes
python research_gate.py --forecast results/ecb/report.json --quotes results/quotes/report.json
```

Файл `raw_eurusd.csv` потрібно отримати самостійно з правомірного джерела котирувань. ECB fixings не є виконуваними цінами. Комісії, спред і параметризоване прослизання враховані, але фінансування, свопи, гарантоване виконання та портфельний ризик — ні.

## Структура

`eialfx.py` — прогнозування з walk-forward; `quote_backtest.py` — симуляція угод; `research_gate.py` — критерії переходу до наступного етапу; `tests/` і GitHub Actions — перевірки; `VARIABLES_AND_SOURCES.csv` — каталог історичних показників.

Вивчи [методологію](docs/RESEARCH_PROTOCOL.md) і [план збору історичних даних](docs/HISTORICAL_DATA_AND_BACKTEST.md). Код окремий від Varrior Labs FX Analytics; жодні історичні результати чи приватні тики автоматично не переносяться.

## Ліцензія

© 2026 Paweł Maruszczyk. [Коротка власницька ліцензія](LICENSE). Перегляд та оцінювання коду дозволені; інше використання потребує письмового дозволу. Не є фінансовою порадою.
