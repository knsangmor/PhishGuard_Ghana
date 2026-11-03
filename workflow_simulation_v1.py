"""Local research workflow with fixed dummy data. No passwords, live sites or training.

--demo and --self-test use Python's standard library only.
--run is a separate exploratory benchmark of simulated pages, not real phishing.
"""
import argparse
import csv
import hashlib
import hmac
import http.client
import importlib.util
import json
import math
import random
import secrets
import threading
import time
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit, urlunsplit

DUMMY_CODE = 'DEMO123'
FIXTURE_SEED = 7042
ROUNDS = [['brand_query', 'pct_1', 'path_32', 'fragment'],
          ['path_256', 'path_512', 'unicode_query', 'long_query'],
          ['pct_2', 'path_96', 'brand_query', 'unicode_query'],
          ['path_512', 'pct_4', 'long_query', 'fragment']]
LIMITATIONS = [
    'Fictional pages and simulated labels; not real phishing or natural traffic.',
    'GET and dummy POST preservation cover only the declared local application.',
    'Directory aliases and ignored query/fragment fields are intentional application policies.',
    'No real authentication, credential capture, payment, delivery campaign or legitimate-provider compromise is tested.',
    'Follow-up designed after examining the earlier controlled-fixture outcomes; exploratory evidence.',
    'Different model representations and unique-query counts prevent architecture-only claims.',
    'Repeated fits on fixed cases do not measure uncertainty over real attacks.'
]
CSS = '''body{margin:0;background:#f2f5f8;color:#172b3a;font:17px/1.6 system-ui,sans-serif}
main{max-width:760px;margin:48px auto;padding:32px;background:white;border-radius:16px}
h1{font-size:30px;line-height:1.2} .badge{color:#22556b;font-weight:700;font-size:14px}
.note{background:#edf5f8;padding:16px;border-radius:8px}button,a.button{background:#13586c;color:white;
border:0;border-radius:7px;padding:12px 18px;font-size:16px;cursor:pointer;text-decoration:none;display:inline-block}
input{padding:10px;font-size:17px;max-width:200px;border:1px solid #9caeb7;border-radius:6px}
label{display:block;margin:18px 0 8px}small{color:#526974}li{margin:12px 0}code{overflow-wrap:anywhere}'''

def digest(data):
    return hashlib.sha256(data).hexdigest()

def page(title, content):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)} | Local research demo</title><style>{CSS}</style></head>'
            '<body><main><p class="badge">LOCAL RESEARCH SIMULATION</p>'
            f'<h1>{escape(title)}</h1>{content}'
            '<p><small>Example Research Services is fictional. Use only the displayed dummy code. '
            'No real account, password or payment is involved.</small></p></main></body></html>').encode()

def fixtures(count=200):
    rng = random.Random(FIXTURE_SEED)
    result = []
    for i in range(count):
        host = f'tenant-{i:03d}.example.test'
        for label, leaf in [(0, 'index.html'), (1, 'login.php')]:
            key = f'{rng.getrandbits(96):024x}'
            route = 'account' if label else 'news'
            result.append({'case_id': key, 'host': host, 'label': label, 'leaf': leaf,
                           'url': f'http://{host}/{route}/{key}/{leaf}?case={key}'})
    return result

def change(url, kind):
    p = urlsplit(url)
    path, query, fragment = p.path, p.query, p.fragment
    if kind.startswith('path_'):
        n = int(kind.split('_')[1])
        path = '/' + ('news/products/support/documentation/' * ((n // 36) + 1))[:n] + '/' + path.lstrip('/')
    elif kind.startswith('pct_'):
        count = int(kind.split('_')[1]); seen = 0; pieces = []; i = 0
        while i < len(path):
            c = path[i]
            if c == '%' and i + 2 < len(path):
                pieces.append(path[i:i + 3]); i += 3; continue
            if c.isascii() and c.isalpha() and seen < count:
                pieces.append(f'%{ord(c):02X}'); seen += 1
            else:
                pieces.append(c)
            i += 1
        path = ''.join(pieces)
    elif kind == 'brand_query': query += '&campaign=mtn-momo-paypal-account'
    elif kind == 'unicode_query': query += '&theme=m\u03bfm\u03bf-support'
    elif kind == 'long_query': query += '&theme=' + 'news-products-support-' * 24
    elif kind == 'fragment': fragment = 'about-products-contact-support'
    else: raise ValueError(kind)
    result = urlunsplit((p.scheme, p.netloc, path, query, fragment))
    return result if len(result) <= 4096 else url

class FormParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.forms = []; self.fields = {}; self.inside = False
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form':
            self.forms.append(a); self.inside = True
        elif tag == 'input' and self.inside and 'name' in a:
            if a['name'] in self.fields: raise ValueError('Repeated form field')
            self.fields[a['name']] = a.get('value', '')
    def handle_endtag(self, tag):
        if tag == 'form': self.inside = False

class WorkflowLab:
    def __init__(self, items, demo=False, port=0):
        self.items = items; self.cases = {x['case_id']: x for x in items}
        self.demo = demo; self.port = port; self.key = secrets.token_bytes(32)
        self.requests = 0; self.successful_submissions = 0
        for case in items:
            case['token'] = hmac.new(self.key, case['case_id'].encode(), hashlib.sha256).hexdigest()
            label = case['label']
            title = 'Example Research Services account demonstration' if label else 'Example Research Services information page'
            text = ('This simulated account prompt asks you to complete a dummy verification.' if label
                    else 'This simulated information page lets you acknowledge that you read the article.')
            button = 'Complete dummy verification' if label else 'Acknowledge demo article'
            case['body'] = page(title, f'<p>{text}</p><p class="note">This is a harmless research exercise. '
                f'The only accepted code is <strong>{DUMMY_CODE}</strong>.</p>'
                '<form method="post" action="">'
                f'<input type="hidden" name="case_id" value="{case["case_id"]}">'
                f'<input type="hidden" name="token" value="{case["token"]}">'
                '<label for="demo_code">Fixed demonstration code</label>'
                f'<input id="demo_code" name="demo_code" value="{DUMMY_CODE}" readonly>'
                f'<p><button type="submit">{button}</button></p></form>')
            case['body_sha256'] = digest(case['body'])
            case['receipt'] = page('Demonstration completed',
                '<p>The local server accepted the fixed dummy code and completed this simulated action.</p>'
                f'<p>Case: <code>{case["case_id"]}</code></p>'
                f'<p>Action: {"dummy account verification" if label else "dummy article acknowledgement"}.</p>')
            case['receipt_sha256'] = digest(case['receipt'])
    @property
    def address(self): return f'127.0.0.1:{self.server.server_port}'
    def route(self, host, target):
        try: p = urlsplit(target); ids = parse_qs(p.query).get('case', [])
        except ValueError: return None
        if len(ids) != 1: return None
        case = self.cases.get(ids[0])
        allowed = case and (host.lower() == case['host'] or (self.demo and host == self.address))
        if not allowed or unquote(p.path).rsplit('/', 1)[-1] != case['leaf']: return None
        return case
    def home(self):
        links = []
        for case in self.items[:2]:
            p = urlsplit(case['url']); path = p.path + '?' + p.query
            label = 'Account-prompt simulation' if case['label'] else 'Information-page simulation'
            links.append(f'<li><a href="{escape(path, quote=True)}">{label}</a></li>')
        positive = next(x for x in self.items if x['label'])
        p = urlsplit(change(positive['url'], 'path_256'))
        links.append(f'<li><a href="{escape(p.path + "?" + p.query, quote=True)}">Account prompt with a long path</a></li>')
        return page('Working URL demonstration', '<p>Open a fictional page and press its button. '
            'Both the ordinary link and changed link should complete the same dummy action.</p>'
            '<ul>' + ''.join(links) + '</ul><p>No detector is running in this viewing mode.</p>')
    def __enter__(self):
        lab = self
        class Handler(BaseHTTPRequestHandler):
            def setup(self): super().setup(); self.connection.settimeout(5)
            def log_message(self, *args): pass
            def reply(self, status, body):
                self.send_response(status)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers(); self.wfile.write(body)
            def do_GET(self):
                lab.requests += 1
                if lab.demo and self.path == '/' and self.headers.get('Host') == lab.address:
                    self.reply(200, lab.home()); return
                case = lab.route(self.headers.get('Host', ''), self.path)
                self.reply(200, case['body']) if case else self.reply(404, b'Unknown demonstration route')
            def do_POST(self):
                lab.requests += 1
                case = lab.route(self.headers.get('Host', ''), self.path)
                if not case: self.reply(404, b'Unknown demonstration route'); return
                origin = self.headers.get('Origin')
                if origin and origin != 'http://' + self.headers.get('Host', ''):
                    self.reply(403, b'Origin rejected'); return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if not 0 < n <= 2048 or self.headers.get('Content-Type', '').split(';')[0] != 'application/x-www-form-urlencoded':
                        raise ValueError('Request shape')
                    fields = parse_qs(self.rfile.read(n).decode('ascii'), keep_blank_values=True, strict_parsing=True)
                except (ValueError, UnicodeError): self.reply(400, b'Invalid dummy submission'); return
                # Only three fixed demonstration fields are ever accepted; no arbitrary input is stored.
                expected = {'case_id': [case['case_id']], 'token': [case['token']], 'demo_code': [DUMMY_CODE]}
                if fields != expected: self.reply(400, b'Only the fixed demonstration fields are accepted'); return
                lab.successful_submissions += 1
                self.reply(200, case['receipt'])
        self.server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start(); return self
    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
    def request(self, case, url, method='GET', fields=None, host=None):
        p = urlsplit(url)
        if p.scheme != 'http' or p.hostname != case['host'] or p.username or p.password or p.port is not None:
            raise ValueError('Only declared logical fixture hosts are allowed')
        target = quote(p.path, safe='/%:@-._~')
        if p.query: target += '?' + quote(p.query, safe='=&%:/?@-._~')
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        headers = {'Host': host or case['host']}
        data = None
        if method == 'POST':
            data = urlencode(fields).encode('ascii'); headers['Content-Type'] = 'application/x-www-form-urlencoded'
        try:
            conn.request(method, target, data, headers)
            r = conn.getresponse(); return r.status, r.getheader('Content-Type'), r.read()
        finally: conn.close()
    def validate(self, case, url):
        try:
            status, content_type, body = self.request(case, url)
            if status != 200 or content_type != 'text/html; charset=utf-8' or digest(body) != case['body_sha256']: return False
            form = FormParser(); form.feed(body.decode())
            if len(form.forms) != 1 or form.forms[0].get('method', '').lower() != 'post' or form.forms[0].get('action') != '': return False
            if form.fields != {'case_id': case['case_id'], 'token': case['token'], 'demo_code': DUMMY_CODE}: return False
            status, content_type, receipt = self.request(case, url, 'POST', form.fields)
            return status == 200 and content_type == 'text/html; charset=utf-8' and digest(receipt) == case['receipt_sha256']
        except (ValueError, OSError, http.client.HTTPException): return False

def self_test():
    items = fixtures(3); checks = 0
    with WorkflowLab(items) as lab:
        for case in items:
            require(lab.validate(case, case['url']), 'Original workflow failed'); checks += 1
            current = case['url']
            for kinds in ROUNDS:
                for kind in kinds:
                    require(lab.validate(case, change(current, kind)), 'Transformation failed: ' + kind); checks += 1
                current = change(current, kinds[0])
            fields = {'case_id': case['case_id'], 'token': case['token'], 'demo_code': DUMMY_CODE}
            for key, bad in [('demo_code', 'WRONG'), ('token', 'wrong'), ('case_id', 'wrong')]:
                payload = dict(fields); payload[key] = bad
                require(lab.request(case, case['url'], 'POST', payload)[0] == 400, 'Invalid form accepted'); checks += 1
            payload = dict(fields, unexpected_field='blocked')
            require(lab.request(case, case['url'], 'POST', payload)[0] == 400, 'Extra field accepted'); checks += 1
            require(not lab.validate(case, case['url'].replace('case=', 'wrong=')), 'Wrong ID accepted'); checks += 1
            require(not lab.validate(case, case['url'].replace(case['leaf'], 'wrong.html')), 'Wrong leaf accepted'); checks += 1
            require(lab.request(case, case['url'], host='wrong.example.test')[0] == 404, 'Wrong tenant accepted'); checks += 1
        require(lab.server.server_address[0] == '127.0.0.1', 'Unexpected interface')
        print(f'SELF-TEST PASSED: {checks} checks; GET pages, parsed forms, fixed-code POST completion, all transformation families and rejection controls.')
        print(f'Loopback HTTP requests: {lab.requests}; successful dummy submissions: {lab.successful_submissions}.')
    print('No models loaded, training, output folders or external-site requests.')

def require(condition, message):
    if not condition: raise ValueError(message)

def write_csv(path, rows):
    if not rows: return
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def backend(root):
    path = root / 'controlled_robustness_v1.py'
    require(path.is_file(), 'Put this file beside the corrected controlled_robustness_v1.py.')
    spec = importlib.util.spec_from_file_location('frozen_v3_scorers', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(hasattr(module, 'verify_saved_models'), 'Use the corrected scorer with verify_saved_models().')
    require(module.ROUNDS == ROUNDS, 'Transformation families differ from the reviewed experiment.')
    # Existing scorer owns the corrected feature replay, feature order, dtype and saved-score checks.
    return module, path

def benchmark(args):
    root = args.root.resolve(); m, scorer_path = backend(root)
    test, files, extract, manifest = m.preflight(root, args.seed)
    print('CHECK PASSED: frozen inputs and corrected features; 400 simulated cases; working dummy GET/POST oracle.')
    if args.check_only:
        print('No model loaded, server, inference, output folders or training.'); return
    if args.verify_only:
        m.verify_saved_models(test, files, extract)
        print('Models verified. No workflow rounds, server, output folders or training.'); return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = root / 'workflow_robustness_v1' / f'seed_{args.seed}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'seed': args.seed, 'fixture_seed': FIXTURE_SEED, 'simulated_cases_per_class': 200,
        'threshold': .5, 'rounds': ROUNDS, 'maximum_new_queries_per_case': 16, 'original_score_queries_separate': True,
        'partition_sha256': m.PART_HASH, 'files': manifest, 'feature_extractor': extract._audit_meta,
        'script_sha256': digest(Path(__file__).read_bytes()), 'scorer_script_sha256': digest(scorer_path.read_bytes()),
        'oracle': 'GET page SHA256 + parsed same-URL form + fixed DEMO123 POST + completion-body SHA256.',
        'routing': 'Correct logical tenant, case query ID and decoded leaf; arbitrary directories; ignores campaign/theme/fragment.',
        'selection': 'Four rounds from current best; minimize simulated-positive score and separately maximize simulated-negative score.',
        'training_performed': False, 'external_test': False, 'exploratory_follow_up': True, 'limitations': LIMITATIONS}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scorers, verification = m.verify_saved_models(test, files, extract)
    (out / 'saved_prediction_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    items = fixtures(); rows = []; trace = []; validations = []; summaries = {}
    with WorkflowLab(items) as lab:
        def validate(case, url, model, round_no, family):
            ok = lab.validate(case, url)
            validations.append({'model': model, 'seed': args.seed, 'case_id': case['case_id'],
                'round': round_no, 'family': family, 'url': url, 'workflow_validated': ok,
                'page_sha256': case['body_sha256'], 'completion_sha256': case['receipt_sha256']})
            return ok
        for name in ['CNN_LSTM', 'RF_full', 'XGB_full']:
            sc = scorers[name]; invalid = skipped = 0
            for case in items: require(validate(case, case['url'], name, 0, 'original'), 'Original dummy workflow failed')
            original = sc.predict([x['url'] for x in items]); best = list(original)
            selected = [x['url'] for x in items]; queries = [0] * len(items)
            seen = [{sc.key(x['url'])} for x in items]
            for round_no, kinds in enumerate(ROUNDS, 1):
                pending = []
                for i, case in enumerate(items):
                    current = selected[i]
                    for kind in kinds:
                        url = change(current, kind)
                        if not validate(case, url, name, round_no, kind): invalid += 1; continue
                        key = sc.key(url)
                        if key in seen[i]: skipped += 1; continue
                        seen[i].add(key); queries[i] += 1; pending.append((i, kind, url))
                scores = sc.predict([u for i, k, u in pending]) if pending else []
                for (i, kind, url), score in zip(pending, scores):
                    trace.append({'model': name, 'seed': args.seed, 'case_id': items[i]['case_id'],
                        'simulated_label': items[i]['label'], 'round': round_no, 'family': kind,
                        'url': url, 'score': score, 'workflow_validated': True,
                        'page_sha256': items[i]['body_sha256'], 'completion_sha256': items[i]['receipt_sha256']})
                    if (items[i]['label'] == 1 and score < best[i]) or (items[i]['label'] == 0 and score > best[i]):
                        best[i] = score; selected[i] = url
                print(f'{name}: round {round_no}/4; new queries {sum(queries):,}', flush=True)
            for i, case in enumerate(items):
                require(validate(case, selected[i], name, 5, 'selected_final'), 'Final selected workflow failed')
                rows.append({'model': name, 'seed': args.seed, 'case_id': case['case_id'],
                    'simulated_label': case['label'], 'original_url': case['url'], 'selected_url': selected[i],
                    'original_score': original[i], 'selected_score': best[i], 'new_queries': queries[i],
                    'page_sha256': case['body_sha256'], 'completion_sha256': case['receipt_sha256']})
            require(max(queries) <= 16, 'Query budget exceeded')
            labels = [x['label'] for x in items]
            den = sum(y == 1 and p >= .5 for y, p in zip(labels, original))
            evaded = sum(y == 1 and p >= .5 and b < .5 for y, p, b in zip(labels, original, best))
            summaries[name] = {'original_fixture_metrics': m.metrics(labels, original),
                'selected_fixture_metrics': m.metrics(labels, best), 'initially_detected_simulated_positives': den,
                'simulated_evasions': evaded, 'conditional_simulated_evasion_rate': evaded / den if den else None,
                'new_score_queries': sum(queries), 'maximum_new_queries_for_one_case': max(queries),
                'invalid_workflows': invalid, 'duplicate_model_inputs_skipped': skipped,
                'saved_prediction_verification': verification[name]}
            for filename, data in [('per_case_results.csv', rows), ('candidate_trace.csv', trace), ('workflow_checks.csv', validations)]:
                write_csv(out / filename, data)
            (out / 'partial_results.json').write_text(json.dumps({'complete': False, 'results': summaries}, indent=2), encoding='utf-8')
        counts = {'loopback_http_requests': lab.requests, 'successful_dummy_submissions': lab.successful_submissions}
    report = {'complete': True, 'protocol': protocol, 'workflow_counts': counts, 'results': summaries}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summaries, indent=2)); print('Saved to:', out)
    print('Working simulated workflows only. No training or changes to existing datasets, models or results.')

def replay_existing(args):
    """Validate previously scored URLs on the new dummy workflow. Never rescore models."""
    source = (args.results or args.root / 'controlled_robustness_v1').resolve()
    require(source.is_dir(), 'Results folder not found: ' + str(source))
    found = {}
    excluded = []
    for path in sorted(source.glob('*/report.json')):
        report = json.loads(path.read_text(encoding='utf-8-sig'))
        if not report.get('complete'): continue
        seed = report.get('protocol', {}).get('seed')
        if seed not in range(42, 47): continue
        if seed in found:
            # Do not choose a repeat by its outcome.
            require(report['results'] == found[seed][1]['results'], 'Repeated seed has different metrics; review it separately.')
            earlier = found[seed][0].parent
            for filename, numeric in [('per_case_results.csv', {'original_score', 'selected_score'}), ('candidate_trace.csv', {'score'})]:
                with (earlier/filename).open(encoding='utf-8-sig', newline='') as f: left = list(csv.DictReader(f))
                with (path.parent/filename).open(encoding='utf-8-sig', newline='') as f: right = list(csv.DictReader(f))
                require(len(left) == len(right), 'Repeated seed has different trace length')
                for a, b in zip(left, right):
                    require(a.keys() == b.keys(), 'Repeated export columns differ')
                    for key in a:
                        if key in numeric:
                            require(abs(float(a[key])-float(b[key])) <= 1e-5 and (float(a[key]) >= .5) == (float(b[key]) >= .5), 'Repeated scores differ beyond tolerance')
                        else: require(a[key] == b[key], 'Repeated case or URL differs; review separately')
            excluded.append(str(earlier))
        found[seed] = (path, report)
    require(set(found) == set(range(42, 47)), 'Expected completed seeds 42–46; no models will be run.')
    items = fixtures(); cases = {x['case_id']: x for x in items}
    reports = []; model_rows = []; failures = []; cache = {}; validation_rows = []
    expected_script = None
    # Audit records are written separately; original results are read only.
    out = args.root.resolve() / 'workflow_preservation_audit' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'operation': 'Retrospective replay of previously scored URLs on a new working dummy application.',
        'new_detector_inference': False, 'training_performed': False, 'seeds': list(range(42, 47)),
        'fixture_seed': FIXTURE_SEED, 'script_sha256': digest(Path(__file__).read_bytes()),
        'oracle': 'GET exact page + parsed form + fixed-code POST + exact completion response.',
        'token_policy': 'Fresh per-run in-memory form tokens; constant for each case within the replay; not part of model inputs.',
        'limitations': LIMITATIONS + ['New workflow checks do not make the earlier pages retrospectively functional.',
            'Existing model scores are reused because the detector inputs are URL strings only; no new detector evaluation was performed.']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    def read_csv(path):
        with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
    def check_score(value):
        v = float(value); require(math.isfinite(v) and 0 <= v <= 1, 'Invalid saved score'); return v
    with WorkflowLab(items) as lab:
        def validate_url(case, url):
            key = (case['case_id'], url)
            if key not in cache:
                ok = lab.validate(case, url); cache[key] = ok
                row = {'case_id': case['case_id'], 'url': url, 'workflow_validated': ok,
                    'page_sha256': case['body_sha256'], 'completion_sha256': case['receipt_sha256']}
                validation_rows.append(row)
                if not ok: failures.append(row)
            return cache[key]
        for seed, (path, report) in sorted(found.items()):
            before = {name: digest((path.parent / name).read_bytes()) for name in
                ['protocol.json', 'report.json', 'per_case_results.csv', 'candidate_trace.csv', 'saved_prediction_verification.json']}
            old_protocol = report['protocol']
            require(old_protocol['fixture_seed'] == FIXTURE_SEED and old_protocol['rounds'] == ROUNDS
                    and old_protocol['threshold'] == .5 and old_protocol['scorer_revision'] == 2,
                    'Existing fixture protocol differs')
            if expected_script is None: expected_script = old_protocol['script_sha256']
            require(old_protocol['script_sha256'] == expected_script, 'Existing scorer changed across seeds')
            checks = json.loads((path.parent / 'saved_prediction_verification.json').read_text())
            rows = read_csv(path.parent / 'per_case_results.csv')
            trace = read_csv(path.parent / 'candidate_trace.csv')
            require(len(rows) == 1200, 'Expected 400 cases for each of three models')
            indexed = {}; histories = {}
            for row in rows:
                key = (row['model'], row['case_id']); require(key not in indexed, 'Duplicate case')
                indexed[key] = row; histories[key] = []
            for row in trace:
                key = (row['model'], row['case_id']); require(key in indexed, 'Trace case missing')
                require(int(row['seed']) == seed and row['simulated_label'] == indexed[key]['simulated_label'], 'Trace alignment')
                require(row['functionality_validated'] == 'True', 'Earlier response oracle did not pass')
                round_no = int(row['round'])
                require(1 <= round_no <= 4 and row['family'] in ROUNDS[round_no - 1], 'Unexpected transformation')
                check_score(row['score']); histories[key].append(row)
            for model in ['CNN_LSTM', 'RF_full', 'XGB_full']:
                group = [r for r in rows if r['model'] == model]
                require(len(group) == 400 and {r['case_id'] for r in group} == set(cases), 'Cases changed')
                require(checks[model]['saved_test_verdict_differences'] == 0
                        and checks[model]['saved_test_score_max_difference'] <= 1e-5, 'Saved model verification failed')
                labels = []; original = []; selected = []; detected = evaded = queries = 0
                for row in group:
                    case = cases[row['case_id']]; label = int(row['simulated_label'])
                    require(int(row['seed']) == seed and label == case['label'] and row['original_url'] == case['url'], 'Case or label mismatch')
                    history = histories[(model, case['case_id'])]
                    require(len(history) == int(row['new_queries']) <= 16, 'Query count mismatch')
                    a = check_score(row['original_score']); best = a; chosen = row['original_url']
                    validate_url(case, row['original_url'])
                    for entry in history:
                        validate_url(case, entry['url']); b = float(entry['score'])
                        if (label == 1 and b < best) or (label == 0 and b > best): best = b; chosen = entry['url']
                    require(best == check_score(row['selected_score']) and chosen == row['selected_url'], 'Selected result mismatch')
                    validate_url(case, row['selected_url'])
                    detected += label == 1 and a >= .5
                    evaded += label == 1 and a >= .5 and best < .5
                    queries += len(history); labels.append(label); original.append(a); selected.append(best)
                def counts(scores):
                    tp = sum(y == 1 and p >= .5 for y, p in zip(labels, scores))
                    fp = sum(y == 0 and p >= .5 for y, p in zip(labels, scores))
                    return {'tp': tp, 'tn': 200-fp, 'fp': fp, 'fn': 200-tp,
                        'recall': tp/200, 'false_positive_rate': fp/200, 'accuracy': (tp+200-fp)/400}
                saved = report['results'][model]
                require(counts(original) == saved['original_fixture_metrics']
                        and counts(selected) == saved['worst_case_fixture_metrics'], 'Saved metrics differ')
                require(detected == saved['originally_detected_simulated_positives'] and evaded == saved['simulated_evasions']
                        and queries == saved['new_score_queries'], 'Saved denominators differ')
                model_rows.append({'seed': seed, 'model': model, 'originally_detected_simulated_positives': detected,
                    'saved_simulated_evasions': evaded, 'new_score_queries_in_old_run': queries,
                    'saved_conditional_evasion_rate': evaded/detected if detected else None})
            for name, value in before.items(): require(digest((path.parent/name).read_bytes()) == value, 'Source file changed during replay')
            reports.append({'seed': seed, 'run': str(path.parent), 'sha256': before})
            print(f'Seed {seed}: saved metrics checked; {len(cache):,} distinct case/URL workflows tested so far.', flush=True)
        counts = {'loopback_http_requests': lab.requests, 'successful_dummy_submissions': lab.successful_submissions}
    write_csv(out/'workflow_checks.csv', validation_rows); write_csv(out/'reused_score_summary.csv', model_rows)
    final = {'complete': True, 'all_workflows_passed': not failures, 'unique_case_url_workflows': len(cache),
        'failed_workflows': len(failures), 'new_detector_inference': False, 'training_performed': False,
        'counts': counts, 'protocol': protocol, 'source_runs': reports, 'excluded_repeat_runs': excluded,
        'incomplete_runs_ignored': [str(p.parent) for p in source.glob('*/protocol.json') if not (p.parent/'report.json').exists()]}
    (out/'report.json').write_text(json.dumps(final, indent=2, allow_nan=False), encoding='utf-8')
    print('WORKFLOW REPLAY COMPLETE:', len(cache), 'distinct case/URL workflows;', len(failures), 'failures.')
    print('Saved to:', out)
    print('Existing scores reused; no training, model inference or source-file changes.')
    require(not failures, 'Some dummy workflows failed; do not claim preservation.')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    for name in ['demo', 'self-test', 'check-only', 'verify-only', 'run', 'audit-existing']: modes.add_argument('--' + name, action='store_true')
    ap.add_argument('--seed', type=int, choices=range(42, 47), default=42)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--results', type=Path, help='Existing controlled_robustness_v1 results folder for --audit-existing.')
    ap.add_argument('--port', type=int, default=0, help='Demo port; 0 chooses an available port.')
    args = ap.parse_args()
    if not 0 <= args.port <= 65535: ap.error('Port must be 0–65535')
    if args.self_test: self_test(); return
    if args.audit_existing: replay_existing(args); return
    if args.demo:
        with WorkflowLab(fixtures(1), demo=True, port=args.port) as lab:
            print('Open this address in your browser:', 'http://' + lab.address + '/', flush=True)
            print('Use only the displayed fixed dummy code. No models run in demo mode.', flush=True)
            print('Press Ctrl+C here to stop.', flush=True)
            try:
                while True: time.sleep(.5)
            except KeyboardInterrupt: print('\nLocal demonstration stopped.')
        return
    benchmark(args)

if __name__ == '__main__': main()
