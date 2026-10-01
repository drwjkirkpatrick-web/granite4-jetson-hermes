#!/usr/bin/env python3
"""
benchmark_granite4.py — quick verification for a running granite4-3b server.

Checks: load, memory, short-gen speed, deep-prompt speed, tool-calling smoke test.

Usage:
    python3 benchmark_granite4.py [--port 8090] [--depth 8000]

Run after `bash launch_granite4.sh 64k`. Writes results JSON next to itself.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request


def http_json(url, payload=None, timeout=600):
    if payload is not None:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_rss():
    r = subprocess.run(['pgrep', '-f', 'granite4-3b.gguf'],
                       capture_output=True, text=True)
    pid = r.stdout.strip().split('\n')[0] if r.stdout.strip() else None
    if not pid:
        return None
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1048576
    except FileNotFoundError:
        return None


def gen(port, prompt, n_tokens, temperature=0.0, timeout=600):
    t0 = time.time()
    resp = http_json(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        {"messages": [{"role": "user", "content": prompt}],
         "max_tokens": n_tokens, "temperature": temperature},
        timeout=timeout)
    dt = time.time() - t0
    u = resp.get('usage', {})
    return u.get('prompt_tokens', 0), u.get('completion_tokens', 0), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--depth', type=int, default=8000,
                    help='approx token depth for the long-prompt test')
    args = ap.parse_args()
    port = args.port

    results: dict = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    # health
    props = http_json(f'http://127.0.0.1:{port}/props', timeout=10)
    results['server'] = {'model': props.get('model_path', '?'), 'port': port}

    # short gen
    pt, ct, dt = gen(port, "Write one paragraph about rivers.", 200)
    results['short_gen'] = {
        'prompt_tokens': pt, 'gen_tokens': ct, 'seconds': round(dt, 1),
        'tok_s': round(ct / dt, 1)}

    # deep prompt
    filler = ("Portland Oregon sits near the confluence of the Willamette and "
              "Columbia rivers. The city grew from a pioneer settlement into a "
              "major timber hub. ") * 1000
    pt, ct, dt = gen(port, filler + "\n\nSummarize the passage in two sentences.", 150)
    results['deep_gen'] = {
        'prompt_tokens': pt, 'gen_tokens': ct, 'seconds': round(dt, 1),
        'pp_tok_s': round(pt / dt, 1),
        'note': 'pp_tok_s approximate (includes generation time)'}

    # tool-calling smoke test (BFCL-style)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city"]}}}]
    payload = {
        "messages": [{"role": "user", "content":
                      "What's the weather in Portland? Use celsius."}],
        "tools": tools, "tool_choice": "auto",
        "max_tokens": 512, "temperature": 0.0}
    t0 = time.time()
    resp = http_json(f'http://127.0.0.1:{port}/v1/chat/completions',
                     payload, timeout=120)
    msg = resp['choices'][0]['message']
    results['tool_call'] = {
        'finish_reason': resp['choices'][0].get('finish_reason'),
        'tool_calls_ok': bool(msg.get('tool_calls')),
        'name': (msg.get('tool_calls') or [{}])[0]
                  .get('function', {}).get('name'),
        'args': (msg.get('tool_calls') or [{}])[0]
                 .get('function', {}).get('arguments')}

    results['server_rss_gb'] = server_rss()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'results', 'benchmark_verify.json')
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()