# Third-party notices

Super_Science is a source-available, proprietary educational project. Its own
code is governed by the repository's `LICENSE`. The third-party component
listed below retains its own license and copyright notice.

## nlohmann/json (version 3.12.0)

- File: `include/json.hpp`
- Project: https://github.com/nlohmann/json
- License: MIT
- Copyright (c) 2013-2025 Niels Lohmann
- Upstream license: https://github.com/nlohmann/json/blob/v3.12.0/LICENSE.MIT

The following MIT notice applies **only** to nlohmann/json; it does not
replace the license governing Super_Science's original code.

```text
MIT License

Copyright (c) 2013-2025 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Development-time tools and optional desktop dependency

`tools/compare_logits.py` and `tools/tokenize_prompt.py` depend on
external Python packages for **offline reference validation only**. No Python
runtime is needed by the core C++ inference programs.

The optional desktop app requires Qt 6 Widgets, which is **not included** in
this repository. Use Qt in accordance with the licensing terms of your
installation.

## Model artifacts

Qwen3 model weights and tokenizer data are **not included** in this repository.
Download model artifacts separately and observe their upstream license terms.
