"""GGUF byte-BPE codec and incremental Qwen chat/tool protocol decoding."""
import json
import math
import re

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers


_QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _fail(message):
    raise ValueError(message)


def _json(value, **kwargs):
    # Transformers' chat-template filter: no HTML escaping or sorted keys.
    return json.dumps(value, ensure_ascii=False, **kwargs)


def _text_content(content):
    if content is None or isinstance(content, str):
        return content or ''
    if isinstance(content, list):
        result = []
        for item in content:
            if (not isinstance(item, dict) or item.get('type') != 'text'
                    or not isinstance(item.get('text'), str)):
                raise ValueError('only text message content is supported')
            result.append(item['text'])
        return ''.join(result)
    raise ValueError('only text message content is supported')


def _tool_schemas(tools):
    if tools is None:
        return {}
    if not isinstance(tools, list):
        raise ValueError('tools must be an array')
    schemas = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            raise ValueError('only function tools are supported')
        function = tool.get('function')
        if not isinstance(function, dict):
            raise ValueError('tool must declare a function')
        name = function.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[^\s<>/=]+', name) or name in schemas:
            raise ValueError('invalid or duplicate function name')
        schema = function.get('parameters', {'type': 'object', 'properties': {}})
        if not isinstance(schema, dict) or schema.get('type', 'object') != 'object':
            raise ValueError('function parameters must be an object schema')
        schemas[name] = schema
    return schemas


class TextCodec:
    """Use tokenizer metadata and the chat template embedded in the GGUF."""

    def __init__(self, metadata):
        if (metadata.get('tokenizer.ggml.model') != 'gpt2'
                or metadata.get('tokenizer.ggml.pre') != 'qwen35'):
            raise ValueError('expected gpt2 tokenizer with qwen35 pre-tokenization')
        tokens = metadata.get('tokenizer.ggml.tokens')
        types = metadata.get('tokenizer.ggml.token_type')
        merges = metadata.get('tokenizer.ggml.merges')
        template = metadata.get('tokenizer.chat_template')
        if (not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens)
                or not isinstance(types, list) or len(types) != len(tokens)
                or not isinstance(merges, list) or not isinstance(template, str) or not template):
            raise ValueError('incomplete GGUF tokenizer metadata')
        if len(set(tokens)) != len(tokens):
            raise ValueError('duplicate tokenizer vocabulary entries')
        if metadata.get('tokenizer.ggml.add_bos_token', False):
            raise ValueError('automatic BOS insertion is not supported for qwen35')
        self._tokens = tokens
        vocab = {token: i for i, token in enumerate(tokens)}
        try:
            pairs = [tuple(merge.split(' ')) for merge in merges]
            self._tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=pairs))
        except Exception as error:
            raise ValueError(f'invalid GGUF BPE vocabulary: {error}') from error
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(_QWEN35_PATTERN), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        self._tokenizer.decoder = decoders.ByteLevel()
        self._special = {i for i, kind in enumerate(types) if kind in (3, 4)}
        self._tokenizer.add_special_tokens([
            AddedToken(tokens[i], special=True, normalized=False) for i in sorted(self._special)
        ])
        self.eos_ids = set()
        for key in ('eos_token_id', 'eot_token_id', 'eom_token_id'):
            token_id = metadata.get('tokenizer.ggml.' + key)
            if token_id is not None:
                if type(token_id) is not int or not 0 <= token_id < len(tokens):
                    raise ValueError('invalid end-of-generation token ID')
                self.eos_ids.add(token_id)
        self.eos_ids.update(vocab[t] for t in ('<|im_end|>', '<|endoftext|>') if t in vocab)
        if not self.eos_ids:
            raise ValueError('missing end-of-generation token IDs')
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        escaped = [b for b in range(256) if b not in visible]
        self._byte_values = {chr(b): b for b in visible}
        self._byte_values.update({chr(256 + i): b for i, b in enumerate(escaped)})
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.filters['tojson'] = _json
        environment.globals['raise_exception'] = _fail
        try:
            self._template = environment.from_string(template)
        except Exception as error:
            raise ValueError(f'invalid GGUF chat template: {error}') from error

    def encode(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self._tokenizer.decode(list(ids), skip_special_tokens=False)

    def token_bytes(self, token_id):
        """Raw bytes for an incremental UTF-8 decoder, including split codepoints."""
        if type(token_id) is not int or not 0 <= token_id < len(self._tokens):
            raise ValueError('token ID out of range')
        token = self._tokens[token_id]
        if token_id in self._special:
            return token.encode('utf-8')
        return bytes(self._byte_values[c] for c in token)

    def render(self, messages, tools=None, enable_thinking=False):
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty array')
        _tool_schemas(tools)
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError('each message must be an object')
            item = dict(message)
            role = item.get('role')
            if role not in ('system', 'developer', 'user', 'assistant', 'tool'):
                raise ValueError('unsupported message role')
            item['role'] = 'system' if role == 'developer' else role
            item['content'] = _text_content(item.get('content'))
            if item.get('tool_calls') is not None:
                if role != 'assistant' or not isinstance(item['tool_calls'], list):
                    raise ValueError('tool_calls must be an assistant array')
                calls = []
                for call in item['tool_calls']:
                    if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                        raise ValueError('invalid assistant function call')
                    function = dict(call['function'])
                    if not isinstance(function.get('name'), str):
                        raise ValueError('function call is missing its name')
                    arguments = function.get('arguments', {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (ValueError, TypeError) as error:
                            raise ValueError('function arguments must be a JSON object') from error
                    if not isinstance(arguments, dict):
                        raise ValueError('function arguments must be a JSON object')
                    function['arguments'] = arguments
                    calls.append(dict(call, function=function))
                item['tool_calls'] = calls
            if item['role'] == 'system' and normalized and all(m['role'] == 'system' for m in normalized):
                normalized[0]['content'] += '\n\n' + item['content']
            else:
                normalized.append(item)
        try:
            return self._template.render(messages=normalized, tools=tools,
                                         enable_thinking=enable_thinking, add_generation_prompt=True)
        except Exception as error:
            raise ValueError(f'cannot render chat template: {error}') from error


def _valid(value, schema):
    """Validate the structural subset needed to safely decode function values."""
    if not isinstance(schema, dict):
        return schema is True
    for keyword in ('anyOf', 'oneOf'):
        if keyword in schema:
            matches = sum(_valid(value, option) for option in schema[keyword])
            if not matches or (keyword == 'oneOf' and matches != 1):
                return False
    if 'enum' in schema and value not in schema['enum']:
        return False
    if 'const' in schema and value != schema['const']:
        return False
    kind = schema.get('type')
    kinds = kind if isinstance(kind, list) else [kind]
    checks = {None: True, 'string': isinstance(value, str), 'boolean': type(value) is bool,
              'integer': type(value) is int, 'number': type(value) in (int, float),
              'object': isinstance(value, dict), 'array': isinstance(value, list), 'null': value is None}
    if not any(checks.get(t, False) for t in kinds):
        return False
    if type(value) is float and not math.isfinite(value):
        return False
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        if any(key not in value for key in schema.get('required', [])):
            return False
        for key, item in value.items():
            if not _valid(item, properties.get(key, schema.get('additionalProperties', True))):
                return False
    if isinstance(value, list) and 'items' in schema:
        if not all(_valid(item, schema['items']) for item in value):
            return False
    return True


def _parameter_value(raw, schema):
    # Remove the template's one framing newline, retaining the string itself.
    if raw.startswith('\r\n'):
        raw = raw[2:]
    elif raw.startswith('\n'):
        raw = raw[1:]
    if raw.endswith('\r\n'):
        raw = raw[:-2]
    elif raw.endswith('\n'):
        raw = raw[:-1]
    if _valid(raw, schema):
        return raw
    try:
        value = json.loads(raw, parse_constant=lambda value: _fail('non-finite tool argument'))
    except ValueError as error:
        raise ValueError('tool argument does not match its declared schema') from error
    if not _valid(value, schema):
        raise ValueError('tool argument does not match its declared schema')
    return value


def _parse_tool(text, schemas, index):
    match = re.fullmatch(r'<tool_call>\s*<function=([^\s<>/=]+)>(.*?)</function>\s*</tool_call>',
                         text, re.DOTALL)
    if not match:
        raise ValueError('malformed tool call')
    name, body = match.groups()
    if name not in schemas:
        raise ValueError(f'unknown function: {name}')
    schema = schemas[name]
    properties = schema.get('properties', {})
    arguments = {}
    while body.strip():
        match = re.match(r'\s*<parameter=([^\s<>/=]+)>(.*?)</parameter>', body, re.DOTALL)
        if not match:
            raise ValueError('malformed function parameter')
        key, raw = match.groups()
        if key in arguments:
            raise ValueError(f'duplicate function parameter: {key}')
        parameter_schema = properties.get(key, schema.get('additionalProperties', True))
        if parameter_schema is False:
            raise ValueError(f'unknown function parameter: {key}')
        arguments[key] = _parameter_value(raw, parameter_schema)
        body = body[match.end():]
    if not _valid(arguments, schema):
        raise ValueError('function arguments do not match their declared schema')
    return {'index': index, 'id': f'call_{index}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}


# Prefixes let us reject malformed control tags before exposing them as content.
_MARKERS = {'<tool_call': '<tool_call>', '</tool_call': '</tool_call>',
            '<think': '<think>', '</think': '</think>',
            '<|im_end': '<|im_end|>', '<|endoftext': '<|endoftext|>',
            '<function=': '<function=', '</function': '</function>',
            '<parameter=': '<parameter=', '</parameter': '</parameter>'}


class DeltaParser:
    """Emit ordinary text promptly; buffer only unfinished protocol markers/calls."""

    def __init__(self, tools=None):
        self._schemas = _tool_schemas(tools)
        self._pending = ''
        self._mode = 'content'
        self._call_count = 0
        self._finished = False

    def feed(self, text):
        if self._finished:
            raise ValueError('assistant parser is already finished')
        self._pending += text
        deltas = []
        while self._pending:
            if self._mode == 'done':
                if self._pending.strip():
                    raise ValueError('text after end-of-generation marker')
                self._pending = ''
                break
            if self._mode == 'tool':
                end = self._pending.find('</tool_call>')
                if end < 0:
                    break
                end += len('</tool_call>')
                call = _parse_tool(self._pending[:end], self._schemas, self._call_count)
                deltas.append({'tool_calls': [call]})
                self._call_count += 1
                self._pending = self._pending[end:]
                self._mode = 'content'
                continue
            positions = [(self._pending.find(prefix), marker) for prefix, marker in _MARKERS.items()
                         if prefix in self._pending]
            if not positions:
                keep = max((size for marker in _MARKERS.values()
                            for size in range(1, min(len(marker), len(self._pending) + 1))
                            if self._pending.endswith(marker[:size])), default=0)
                count = len(self._pending) - keep
                if count:
                    deltas.append({self._mode: self._pending[:count]})
                    self._pending = self._pending[count:]
                break
            index, marker = min(positions)
            if index:
                deltas.append({self._mode: self._pending[:index]})
                self._pending = self._pending[index:]
            if marker.startswith(self._pending) and len(self._pending) < len(marker):
                break
            if not self._pending.startswith(marker):
                raise ValueError('malformed assistant control marker')
            if marker == '<tool_call>':
                if self._mode != 'content':
                    raise ValueError('tool call inside reasoning')
                self._mode = 'tool'
                continue
            self._pending = self._pending[len(marker):]
            if marker == '<think>' and self._mode == 'content':
                self._mode = 'reasoning_content'
            elif marker == '</think>' and self._mode == 'reasoning_content':
                self._mode = 'content'
            elif marker in ('<|im_end|>', '<|endoftext|>') and self._mode == 'content':
                self._mode = 'done'
            else:
                raise ValueError('unexpected assistant control marker')
        return deltas

    def finish(self):
        deltas = self.feed('')
        if self._mode not in ('content', 'reasoning_content', 'done'):
            raise ValueError('truncated assistant control block')
        if self._pending:
            if len(self._pending) > 1:
                raise ValueError('truncated assistant control marker')
            deltas.append({self._mode: self._pending})
            self._pending = ''
        self._finished = True
        return deltas


def parse_assistant(text, tools=None):
    parser = DeltaParser(tools)
    deltas = parser.feed(text) + parser.finish()
    message = {'role': 'assistant', 'content': None}
    for delta in deltas:
        for field in ('content', 'reasoning_content'):
            if field in delta:
                message[field] = (message.get(field) or '') + delta[field]
        for call in delta.get('tool_calls', []):
            call = dict(call)
            call.pop('index')
            message.setdefault('tool_calls', []).append(call)
    return message
