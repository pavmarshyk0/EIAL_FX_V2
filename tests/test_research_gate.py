from research_gate import evaluate


def _forecast(label='HISTORICAL REFERENCE-RATE RESEARCH - NOT EXECUTABLE PNL'):
    metric = {'n_test': 400, 'oos_r2_vs_zero': .01, 'rmse': .1}
    baseline = {'n_test': 400, 'oos_r2_vs_zero': -.1, 'rmse': .12}
    return {'label': label, 'metrics': {'all_observed_factors': {'eurusd_next_logret': metric, 'eurpln_next_logret': metric},
                                       'price_baseline': {'eurusd_next_logret': baseline, 'eurpln_next_logret': baseline}}}


def _quotes(label='EXECUTABLE QUOTE PAPER SIMULATION - VERIFY SOURCE'):
    return {'label': label, 'pairs': {'eurusd': {'n_trades': 150, 'mean_net_bps_per_trade': 0.6}}}


def test_screen_never_proves_alpha():
    r = evaluate(_forecast(), _quotes())
    assert r['status'] == 'FURTHER_VALIDATION_REQUIRED'
    assert r['edge_proven'] is False


def test_synthetic_rejected_even_if_metrics_good():
    r = evaluate(_forecast('SYNTHETIC DEMONSTRATION'), _quotes('SYNTHETIC QUOTES'))
    assert r['status'] == 'NO_GO_SCREEN'
    assert 'synthetic_forecast_data' in r['reason_codes']
    assert 'synthetic_quote_data' in r['reason_codes']


def test_negative_improvement_rejected():
    f = _forecast()
    f['metrics']['all_observed_factors']['eurusd_next_logret'] = {'n_test': 400, 'oos_r2_vs_zero': -0.1, 'rmse': 0.2}
    r = evaluate(f, _quotes())
    assert r['status'] == 'NO_GO_SCREEN'
