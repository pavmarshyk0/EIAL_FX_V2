[English](README.md) | [Українська](README.uk.md) | **Polski**

# EIAL-FX — badania nad informacyjno-algorytmicznymi czynnikami FX

**Status: `NO_EDGE_ESTABLISHED`.** To eksperymentalny projekt badawczy w Pythonie, a nie sprawdzona strategia inwestycyjna. Brak potwierdzonej przewagi na rzeczywistych danych; przykłady demonstracyjne wykorzystują wyłącznie dane syntetyczne.

Model analizuje EUR/USD i EUR/PLN. USD/PLN jest wyliczany z relacji trójkątnej. Czynniki: opóźnione stopy procentowe, historia kursów, dostępne dane instytucjonalne, sentyment i mikrostruktura. Każda zmienna ma wymagać **znacznika rzeczywistej dostępności informacji**, aby uniknąć przecieku danych z przyszłości.

## Instalacja i test

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python eialfx.py demo --out results/demo
python quote_backtest.py demo --out results/quote_demo
```

**Historyczne kursy referencyjne ECB:**

```bash
python eialfx.py fetch-ecb --start 2016-01-01 --out data/ecb_fx.csv
python eialfx.py train --rates data/ecb_fx.csv --out results/ecb
```

**Paper backtest na rzeczywistych kwotowaniach bid/ask:**

```bash
python dukascopy_to_quotes.py --input raw_eurusd.csv --pair eurusd --out data/eurusd_quotes.csv
python quote_backtest.py run --predictions results/ecb/oos_predictions.csv --rates data/ecb_fx.csv --quotes data/eurusd_quotes.csv --out results/quotes
python research_gate.py --forecast results/ecb/report.json --quotes results/quotes/report.json
```

Plik `raw_eurusd.csv` należy pozyskać z legalnego źródła danych. Kursy referencyjne ECB nie są wykonywalnymi cenami transakcyjnymi. Backtester uwzględnia spread, poślizg i opłaty, ale nie modeluje finansowania, swapów, odrzuceń zleceń ani zarządzania pełnym portfelem.

## Pliki i metoda

`eialfx.py` — walidacja walk-forward; `quote_backtest.py` — symulacja transakcji; `research_gate.py` — weryfikacja kryteriów eksperymentu; `tests/` i GitHub Actions — testy automatyczne; `VARIABLES_AND_SOURCES.csv` — katalog źródeł.

Zobacz [protokół badawczy](docs/RESEARCH_PROTOCOL.md) i [pozyskiwanie danych](docs/HISTORICAL_DATA_AND_BACKTEST.md). EIAL-FX jest niezależnym projektem od Varrior Labs FX Analytics; nie dziedziczy prywatnych danych ani zweryfikowanych wyników.

## Licencja

© 2026 Paweł Maruszczyk. [Krótka licencja własnościowa](LICENSE). Kod można przeglądać w celu oceny; dalsze wykorzystanie wymaga pisemnej zgody. Nie stanowi porady inwestycyjnej.

## AI Alpha Discovery v0.1 — eksperymentalne poszukiwanie hipotez

Moduł `alpha_discovery/` generuje i weryfikuje hipotezy dotyczące EUR/USD oraz EUR/PLN. Opcjonalny lokalny model Ollama (`--llm-model gemma3:4b`) proponuje wyłącznie dozwolone specyfikacje JSON — żaden wygenerowany kod Pythona nie jest uruchamiany. Każdy czynnik jest porównywany z modelem bazowym na chronologicznych danych, a wszystkie próby (także nieudane) są zapisywane. MLP to jedynie opcjonalny eksperyment.

```bash
python -m alpha_discovery --demo --out results/alpha_demo
python -m alpha_discovery --rates data/ecb_fx.csv --dev-end 2025-12-31 --out results/alpha_real
```

**Ograniczenie:** kursy referencyjne ECB nie są cenami wykonania zleceń. Dane za 2026 r. były już analizowane i nie stanowią ślepej próby. Nie potwierdzono dotąd przewagi inwestycyjnej. Dokumentacja: [`docs/AI_ALPHA_DISCOVERY.md`](docs/AI_ALPHA_DISCOVERY.md).
