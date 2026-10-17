"""Gemini Robotics-ER による「次の一手」の決定（ROSに依存しない）.

観察 → 考える → 動く を1手ずつ繰り返す. 毎回最新の画像を見せるので、
掴み損ねた・ずれて置いた等の失敗も次の手で自然にやり直せる.
"""

from dataclasses import dataclass, field
import json
import os
import re
import time

DEFAULT_MODEL_ID = 'gemini-robotics-er-2-preview'
# 主モデルの1日の無料枠を使い切ったときに順に切り替えるモデル（無料枠はモデルごとに別）
DEFAULT_FALLBACK_MODEL_IDS = ('gemini-3.5-flash', 'gemini-3-flash-preview',
                              'gemini-3.1-flash-lite', 'gemini-2.5-flash')
DEFAULT_SCENE_HINT = (
    'The table has small solid-colored cubes (5 cm) that can be picked, and larger flat '
    'square trays (8 cm) with a white border and a colored center that serve as destinations.')

NEXT_STEP_PROMPT = """\
You are the high-level brain of a small robot arm (CRANE+ V2) working on a table.
The image is taken by a camera looking straight down at the table.
{scene_hint}
The arm can do exactly one primitive: pick ONE object and place it at ONE location.

User instruction (may be Japanese): "{instruction}"

Steps already executed (oldest first). Check the image to see whether they actually succeeded:
{history}

Decide the single NEXT pick-and-place step needed to fulfil the instruction,
or report that the instruction is already fulfilled (or impossible).

Rules:
- Points are [y, x] normalized to 0-1000 (y down, x right).
- "pick.point" must be the center of the top face of the object to grasp.
- "place.point" must be the center of a FREE area on the destination (not on top of another object).
- Only pick small cubes. Never pick trays, the robot, or the camera stand.
- If a previous step failed (object not at destination), you may retry it.

Answer ONLY with JSON in this exact schema:
{{
  "done": false,
  "pick": {{"label": "red cube", "point": [y, x]}},
  "place": {{"label": "blue tray", "point": [y, x]}},
  "reason": "short English reason (shown on video)",
  "say_ja": "短い日本語の説明"
}}
When done (or impossible), answer {{"done": true, "reason": "...", "say_ja": "..."}}.
"""

POINT_PROMPT = """\
Point to {query} in the image.
Answer ONLY with JSON: [{{"point": [y, x], "label": "<name>"}}, ...]
Points are [y, x] normalized to 0-1000.
"""


@dataclass
class Target:
    label: str
    point_yx: tuple


@dataclass
class StepDecision:
    done: bool
    pick: Target = None
    place: Target = None
    reason: str = ''
    say_ja: str = ''
    raw: dict = field(default_factory=dict)


def extract_json(text):
    """```json ... ``` のコードフェンスや前後の文章が付いていても JSON 部分を取り出す."""
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find('{'), text.find('[')) if i >= 0]
    if not starts:
        raise ValueError(f'no JSON found in response: {text!r}')
    start = min(starts)
    end = max(text.rfind('}'), text.rfind(']'))
    return json.loads(text[start:end + 1])


def _parse_target(obj, name):
    if not isinstance(obj, dict):
        raise ValueError(f'"{name}" must be an object, got {obj!r}')
    point = obj.get('point')
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or not all(isinstance(p, (int, float)) for p in point)):
        raise ValueError(f'"{name}.point" must be [y, x], got {point!r}')
    if not all(0 <= p <= 1000 for p in point):
        raise ValueError(f'"{name}.point" out of 0-1000 range: {point!r}')
    return Target(label=str(obj.get('label', name)), point_yx=(float(point[0]), float(point[1])))


def parse_step_decision(data):
    """Gemini の JSON 応答を StepDecision に変換する. 不正なら ValueError."""
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f'expected a JSON object, got {data!r}')
    reason = str(data.get('reason', ''))
    say_ja = str(data.get('say_ja', ''))
    if bool(data.get('done', False)):
        return StepDecision(done=True, reason=reason, say_ja=say_ja, raw=data)
    return StepDecision(
        done=False,
        pick=_parse_target(data.get('pick'), 'pick'),
        place=_parse_target(data.get('place'), 'place'),
        reason=reason,
        say_ja=say_ja,
        raw=data,
    )


def format_history(history):
    if not history:
        return '(none yet)'
    return '\n'.join(f'{i + 1}. {h}' for i, h in enumerate(history))


class GeminiBrain:
    def __init__(self, model_id=DEFAULT_MODEL_ID, api_key=None,
                 thinking_budget=1024, temperature=0.5, scene_hint=DEFAULT_SCENE_HINT,
                 fallback_model_ids=DEFAULT_FALLBACK_MODEL_IDS, log=print):
        from google import genai
        from google.genai import types

        api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY が設定されていません')
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self._fallbacks = [m for m in fallback_model_ids if m and m != model_id]
        self._log = log
        self.scene_hint = scene_hint
        self._config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )

    def _switch_model(self, why):
        old, self.model_id = self.model_id, self._fallbacks.pop(0)
        self._log(f'{why} for {old}; switching to {self.model_id}')

    def _ask(self, jpeg_bytes, prompt, max_attempts=6, overload_attempts=3):
        """max_attempts はモデルごとの再試行回数. 混雑が overload_attempts 回続いたら次のモデルへ."""
        from google.genai import errors

        image = self._types.Part.from_bytes(data=jpeg_bytes, mime_type='image/jpeg')
        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.model_id, contents=[image, prompt], config=self._config)
                return response.text
            except errors.APIError as e:
                if e.code == 429 and 'PerDay' in str(e):
                    # 1日の無料枠切れは待っても戻らないので、別モデルに切り替える
                    if not self._fallbacks:
                        raise RuntimeError(
                            f'{self.model_id} の1日の無料枠を使い切りました'
                            '（課金を有効にするか、翌日まで待ってください）') from e
                    self._switch_model('daily quota exhausted')
                    attempt = 0
                    continue
                # 混雑 (503) や分単位のレート制限 (429) は待てば通るので指数バックオフで再試行
                if e.code not in (429, 500, 503):
                    raise
                attempt += 1
                if e.code in (500, 503) and attempt >= overload_attempts and self._fallbacks:
                    # 混雑が続くモデルは見切って次のモデルへ（撮影を途中で止めないため）
                    self._switch_model(f'overloaded ({e.code})')
                    attempt = 0
                    continue
                if attempt >= max_attempts:
                    raise
                time.sleep(min(2 ** attempt, 30))

    def next_step(self, jpeg_bytes, instruction, history, retries=2):
        """次の1手を決める. JSON が壊れていたら retries 回まで聞き直す."""
        prompt = NEXT_STEP_PROMPT.format(
            instruction=instruction, history=format_history(history),
            scene_hint=self.scene_hint)
        last_error = None
        for _ in range(retries + 1):
            text = self._ask(jpeg_bytes, prompt)
            try:
                return parse_step_decision(extract_json(text))
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
        raise RuntimeError(f'Gemini の応答を解釈できませんでした: {last_error}')

    def point(self, jpeg_bytes, query):
        """デバッグ用: 任意の物体を指し示させる."""
        prompt = POINT_PROMPT.format(query=query, scene_hint=self.scene_hint)
        data = extract_json(self._ask(jpeg_bytes, prompt))
        return [_parse_target(d, 'point') for d in data]
