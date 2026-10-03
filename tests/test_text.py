"""CPU-only tokenizer and chat protocol tests."""
import codecs
import json
import os
from pathlib import Path
import unittest

from qvelox.text import DeltaParser, TextCodec, parse_assistant

# Extracted verbatim from tokenizer.chat_template in the target Qwen GGUF.
TEMPLATE = r'''{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}'''


def tiny_metadata():
    # IDs 0..255 intentionally equal byte values, independently of GPT-2 order.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    escaped = [b for b in range(256) if b not in visible]
    alphabet = {b: chr(b) for b in visible}
    alphabet.update({b: chr(256 + i) for i, b in enumerate(escaped)})
    tokens = [alphabet[b] for b in range(256)] + ['ab', 'abc', '12',
        '<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>']
    return {'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
            'tokenizer.ggml.tokens': tokens,
            'tokenizer.ggml.token_type': [1] * 259 + [3, 3, 3, 4, 4],
            'tokenizer.ggml.merges': ['a b', 'ab c', '1 2'],
            'tokenizer.ggml.eos_token_id': 260,
            'tokenizer.ggml.add_bos_token': False,
            'tokenizer.chat_template': TEMPLATE}


TOOLS = [{'type': 'function', 'function': {'name': 'write', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'},
        'count': {'type': 'integer'}, 'enabled': {'type': 'boolean'},
        'data': {'type': 'object'}, 'items': {'type': 'array'},
        'nothing': {'type': 'null'}},
    'required': ['path'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'ping', 'parameters': {
        'type': 'object', 'properties': {}, 'additionalProperties': False}}}]


def collect(deltas):
    result = {'role': 'assistant', 'content': None}
    for delta in deltas:
        for field in ('content', 'reasoning_content'):
            if field in delta:
                result[field] = (result.get(field) or '') + delta[field]
        for call in delta.get('tool_calls', []):
            call = dict(call)
            call.pop('index')
            result.setdefault('tool_calls', []).append(call)
    return result


class TextCodecTests(unittest.TestCase):
    def setUp(self):
        self.codec = TextCodec(tiny_metadata())

    def test_byte_ids_merges_numeric_split_and_special_tokens(self):
        self.assertEqual(self.codec.encode('abc 123é🙂<|im_end|>'),
                         [257, 32, 49, 50, 51, 195, 169, 240, 159, 153, 130, 260])
        self.assertEqual(self.codec.decode([257, 32, 195, 169, 260]), 'abc é<|im_end|>')
        self.assertEqual(self.codec.eos_ids, {260, 261})

    def test_individual_byte_pieces_support_unicode_streaming(self):
        decoder = codecs.getincrementaldecoder('utf-8')()
        chunks = [decoder.decode(self.codec.token_bytes(i)) for i in [240, 159, 153, 130, 260]]
        self.assertEqual(chunks, ['', '', '', '🙂', '<|im_end|>'])
        self.assertEqual(self.codec.token_bytes(257), b'abc')

    def test_official_template_and_developer_system_combination(self):
        prompt = self.codec.render([{'role': 'system', 'content': 'First'},
                                   {'role': 'developer', 'content': 'Second'},
                                   {'role': 'user', 'content': [{'type': 'text', 'text': 'Ciao é'}]}])
        self.assertEqual(prompt, '<|im_start|>system\nFirst\n\nSecond<|im_end|>\n'
                         '<|im_start|>user\nCiao é<|im_end|>\n'
                         '<|im_start|>assistant\n<think>\n\n</think>\n\n')

    def test_tool_history_json_arguments_and_template_tojson(self):
        messages = [{'role': 'user', 'content': 'Write'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'a', 'type': 'function',
                'function': {'name': 'write', 'arguments': '{"path":"a<b&é","count":2}'}}]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'ok'}]
        before = json.dumps(messages)
        prompt = self.codec.render(messages, TOOLS)
        self.assertIn(json.dumps(TOOLS[0], ensure_ascii=False), prompt)
        self.assertIn('<parameter=path>\na<b&é\n</parameter>', prompt)
        self.assertIn('<parameter=count>\n2\n</parameter>', prompt)
        self.assertIn('<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>', prompt)
        self.assertEqual(json.dumps(messages), before)
        thinking = self.codec.render([{'role': 'user', 'content': 'why'}], enable_thinking=True)
        self.assertTrue(thinking.endswith('<|im_start|>assistant\n<think>\n'))

    def test_rejects_unsupported_content_roles_and_argument_json(self):
        for messages in ([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}],
                         [{'role': 'user', 'content': {'text': 'bad'}}],
                         [{'role': 'fish', 'content': 'bad'}], []):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.codec.render(messages)
        for arguments in ('not json', '[]', 'null'):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.codec.render([{'role': 'user', 'content': 'x'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'function': {'name': 'ping', 'arguments': arguments}}]}], TOOLS)

    def test_rejects_unsupported_or_incomplete_metadata(self):
        for key, value in [('tokenizer.ggml.pre', 'unknown'),
                           ('tokenizer.ggml.model', 'llama'),
                           ('tokenizer.chat_template', ''),
                           ('tokenizer.ggml.token_type', [1])]:
            metadata = tiny_metadata()
            metadata[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                TextCodec(metadata)


class AssistantParserTests(unittest.TestCase):
    def test_tool_arguments_keep_strings_and_decode_declared_types(self):
        text = '<think>check é</think>Now\n<tool_call>\n<function=write>\n' \
               '<parameter=path>\n 001 true & <tag> \n</parameter>\n' \
               '<parameter=count>2</parameter><parameter=enabled>true</parameter>' \
               '<parameter=data>{"a":3}</parameter><parameter=items>[1,"é"]</parameter>' \
               '<parameter=nothing>null</parameter></function></tool_call>'
        message = parse_assistant(text, TOOLS)
        self.assertEqual(message['content'], 'Now\n')
        self.assertEqual(message['reasoning_content'], 'check é')
        call = message['tool_calls'][0]
        self.assertEqual(call['function']['name'], 'write')
        self.assertEqual(json.loads(call['function']['arguments']), {'path': ' 001 true & <tag> ',
            'count': 2, 'enabled': True, 'data': {'a': 3}, 'items': [1, 'é'], 'nothing': None})

    def test_no_arguments_and_multiple_calls(self):
        text = '<tool_call><function=ping></function></tool_call>' * 2
        message = parse_assistant(text, TOOLS)
        self.assertIsNone(message['content'])
        self.assertEqual([c['function']['arguments'] for c in message['tool_calls']], ['{}', '{}'])
        self.assertEqual(len({c['id'] for c in message['tool_calls']}), 2)

    def test_streaming_is_independent_of_every_character_boundary(self):
        text = 'Ciao 🙂 < 3<think>réfléchir</think>\n<tool_call><function=write>' \
               '<parameter=path>é.txt</parameter></function></tool_call><|im_end|>'
        expected = parse_assistant(text, TOOLS)
        for size in (1, 2, 3, 7, 13, len(text)):
            parser = DeltaParser(TOOLS)
            deltas = []
            for i in range(0, len(text), size):
                deltas.extend(parser.feed(text[i:i + size]))
            deltas.extend(parser.finish())
            with self.subTest(size=size):
                self.assertEqual(collect(deltas), expected)
                self.assertFalse(any('<tool_call>' in d.get('content', '') for d in deltas))

    def test_regular_content_streams_immediately_and_partial_markers_wait(self):
        parser = DeltaParser(TOOLS)
        self.assertEqual(parser.feed('hello 🙂'), [{'content': 'hello 🙂'}])
        self.assertEqual(parser.feed('<tool_ca'), [])
        self.assertEqual(parser.feed('ll><function=ping></function></tool_call>')[0]
                         ['tool_calls'][0]['function'], {'name': 'ping', 'arguments': '{}'})
        self.assertEqual(parser.finish(), [])

    def test_malformed_unknown_truncated_or_invalid_calls_raise(self):
        cases = ['<tool_call><function=unknown></function></tool_call>',
                 '<tool_call><function=ping></tool_call>',
                 '<tool_call><function=ping></function>', '<tool_ca',
                 '<tool_call attr=bad><function=ping></function></tool_call>',
                 '<function=ping></function>',
                 '<tool_call><function=write></function></tool_call>',
                 '<tool_call><function=write><parameter=path>x</parameter>'
                 '<parameter=count>true</parameter></function></tool_call>',
                 '<tool_call><function=write><parameter=path>x</parameter>'
                 '<parameter=path>y</parameter></function></tool_call>',
                 '<tool_call><function=write><parameter=path>x</parameter>'
                 '<parameter=bad>y</parameter></function></tool_call>']
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_assistant(text, TOOLS)
        with self.assertRaises(ValueError):
            parse_assistant('<tool_call><function=ping></function></tool_call>')

    def test_plain_angles_and_eos_never_corrupt_text(self):
        self.assertEqual(parse_assistant('x < y & é'), {'role': 'assistant', 'content': 'x < y & é'})
        self.assertEqual(parse_assistant('done<|im_end|>'), {'role': 'assistant', 'content': 'done'})


@unittest.skipUnless(os.environ.get('REDSHIFT_TOKENIZER_METADATA'), 'real GGUF metadata not supplied')
class ActualTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads(Path(os.environ['REDSHIFT_TOKENIZER_METADATA']).read_text())
        cls.codec = TextCodec(cls.metadata)

    def test_real_special_ids_and_multilingual_roundtrip(self):
        for word in ('<|im_start|>', '<|im_end|>', '<think>', '</think>'):
            self.assertEqual(self.codec.encode(word), [self.metadata['tokenizer.ggml.tokens'].index(word)])
        text = 'Ciao café 中文 🙂\nfor (x=123; x<9; x++) {} <|im_end|>'
        self.assertEqual(self.codec.decode(self.codec.encode(text)), text)


if __name__ == '__main__':
    unittest.main()
