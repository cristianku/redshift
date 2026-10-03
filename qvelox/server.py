"""Serve the Redshift CUDA runtime through the OpenAI Chat Completions API."""
import argparse
import codecs
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import random
import select
import socket
import threading
import time
import uuid

MODEL_ID = 'redshift-qwen3.8-27b'
MAX_BODY = 1024 * 1024
LOG = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, message, status=400, code='invalid_request_error'):
        super().__init__(message)
        self.status, self.code = status, code

    def payload(self):
        return {'error': {'message': str(self), 'type': self.code, 'code': self.code}}


class Cancelled(Exception):
    pass


@dataclass
class Request:
    prompt: list
    max_tokens: int
    temperature: float
    top_p: float
    seed: int | None
    stream: bool
    include_usage: bool
    tools: list | None
    thinking: bool
    stop: list
    parallel_tools: bool


def number(data, name, default, lower, upper, *, exclusive_lower=False):
    value = data.get(name, default)
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value > upper or (value <= lower if exclusive_lower else value < lower)):
        raise APIError(f'{name} must be a finite number in {"(" if exclusive_lower else "["}{lower}, {upper}]')
    return value


class Engine:
    """One runtime owner; the HTTP handler holds lock for an entire generation."""
    def __init__(self, runtime, codec, model_id=MODEL_ID, context=139264):
        self.runtime, self.codec = runtime, codec
        self.model_id, self.context = model_id, context
        if type(context) is not int or context < 1:
            raise ValueError('context must be a positive integer')
        if getattr(runtime, 'context', context) != context:
            raise ValueError('HTTP context must match runtime capacity')
        self.lock = threading.Lock()
        self.ready = True
        self.resident = []
        self.prediction = None
        self.prediction_is_logits = False
        self.created = int(time.time())

    @property
    def busy(self):
        return self.lock.locked()

    def prepare(self, data):
        if not isinstance(data, dict):
            raise APIError('request body must be a JSON object')
        if data.get('model') != self.model_id:
            raise APIError(f'model must be {self.model_id}', 404, 'model_not_found')
        for key in ('stream', 'enable_thinking', 'parallel_tool_calls'):
            if key in data and type(data[key]) is not bool:
                raise APIError(f'{key} must be a boolean')
        if type(data.get('n', 1)) is not int or data.get('n', 1) != 1:
            raise APIError('only n=1 is supported')
        temperature = number(data, 'temperature', 0, 0, 2)
        top_p = number(data, 'top_p', 1, 0, 1, exclusive_lower=True)
        for key in ('frequency_penalty', 'presence_penalty'):
            if number(data, key, 0, -2, 2) != 0:
                raise APIError(f'{key} is not supported; use 0')
        for key in ('logit_bias', 'logprobs', 'top_logprobs', 'functions', 'function_call',
                    'audio', 'prediction', 'response_schema'):
            if data.get(key):
                raise APIError(f'{key} is not supported')
        if data.get('modalities', ['text']) != ['text']:
            raise APIError('only text modality is supported')
        if data.get('response_format', {'type': 'text'}) != {'type': 'text'}:
            raise APIError('only response_format type=text is supported')
        seed = data.get('seed')
        if seed is not None and type(seed) is not int:
            raise APIError('seed must be an integer')
        options = data.get('stream_options')
        if options is None:
            options = {}
        if not isinstance(options, dict) or type(options.get('include_usage', False)) is not bool:
            raise APIError('stream_options.include_usage must be a boolean')
        tools = data.get('tools')
        if tools is not None and not isinstance(tools, list):
            raise APIError('tools must be an array')
        choice = data.get('tool_choice', 'auto')
        if choice not in ('auto', 'none'):
            raise APIError('only tool_choice auto or none is supported')
        if choice == 'none':
            tools = None
        stop = data.get('stop')
        if stop is None:
            stop = []
        if isinstance(stop, str):
            stop = [stop]
        if (not isinstance(stop, list) or len(stop) > 4
                or any(not isinstance(s, str) or not s or len(s) > 1024 for s in stop)):
            raise APIError('stop must be a string or up to four nonempty strings of at most 1024 characters')
        thinking = data.get('enable_thinking', False)
        try:
            prompt = self.codec.encode(self.codec.render(data.get('messages'), tools=tools,
                                                       enable_thinking=thinking))
        except (ValueError, TypeError, KeyError) as error:
            raise APIError(str(error)) from error
        if not prompt:
            raise APIError('rendered prompt has no tokens')
        remaining = self.context - len(prompt)
        if remaining < 1:
            raise APIError(f'prompt has {len(prompt)} tokens; context capacity is {self.context} and needs room for completion',
                           code='context_length_exceeded')
        maximum = data.get('max_completion_tokens', data.get('max_tokens', min(512, remaining)))
        if 'max_completion_tokens' in data and 'max_tokens' in data and data['max_tokens'] != maximum:
            raise APIError('max_tokens and max_completion_tokens disagree')
        if type(maximum) is not int or maximum < 1:
            raise APIError('max_tokens/max_completion_tokens must be a positive integer')
        if maximum > remaining:
            raise APIError(f'prompt has {len(prompt)} tokens and requested completion {maximum}; '
                           f'context capacity is {self.context}, leaving {remaining} completion tokens',
                           code='context_length_exceeded')
        return Request(prompt, maximum, temperature, top_p, seed, data.get('stream', False),
                       options.get('include_usage', False), tools, thinking, stop,
                       data.get('parallel_tool_calls', True))

    def reset(self):
        self.resident = []
        self.prediction = None
        self.runtime.reset()

    def recover(self):
        try:
            self.reset()
        except Exception:
            self.ready = False
            LOG.error('Runtime reset failed; engine unavailable')

    def forward(self, tokens, logits=False):
        rows = self.runtime.evaluate(tokens) if logits else self.runtime.advance(tokens)
        if len(rows) != len(tokens):
            raise RuntimeError('runtime returned an invalid prediction count')
        self.resident.extend(tokens)
        self.prediction = rows[-1]
        self.prediction_is_logits = logits
        if self.runtime.position != len(self.resident):
            raise RuntimeError('runtime position differs from resident token count')

    def prefill(self, request, check):
        if self.runtime.position != len(self.resident):
            self.reset()
        # DeltaNet state is recurrent: only the complete resident prefix can be reused.
        reusable = request.prompt[:len(self.resident)] == self.resident
        needs_logits = request.temperature > 0
        if (not reusable or (len(request.prompt) == len(self.resident)
                             and needs_logits and not self.prediction_is_logits)):
            self.reset()
        offset = len(self.resident)
        while offset < len(request.prompt):
            check()
            chunk = request.prompt[offset:offset + 8]
            self.forward(chunk, logits=needs_logits and offset + len(chunk) == len(request.prompt))
            offset += len(chunk)
        check()

    def sample(self, request, rng):
        if not self.prediction_is_logits:
            return self.prediction
        logits = self.prediction
        if not logits or not all(math.isfinite(x) for x in logits):
            raise RuntimeError('runtime produced invalid logits')
        if request.temperature == 0:
            return max(range(len(logits)), key=logits.__getitem__)
        highest = max(logits)
        weights = [math.exp((x - highest) / request.temperature) for x in logits]
        indices = list(range(len(weights)))
        if request.top_p < 1:
            indices.sort(key=weights.__getitem__, reverse=True)
            threshold = request.top_p * sum(weights)
            cumulative = 0.0
            for count, index in enumerate(indices, 1):
                cumulative += weights[index]
                if cumulative >= threshold:
                    indices = indices[:count]
                    break
        return rng.choices(indices, weights=[weights[index] for index in indices], k=1)[0]

    def generate(self, request, check, emit):
        from .text import DeltaParser, parse_assistant
        self.prefill(request, check)
        rng = random.Random(request.seed)
        parser = DeltaParser(tools=request.tools)
        if request.thinking:
            for delta in parser.feed('<think>'):
                emit(delta)
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        generated, decoded, visible = [], '', ''
        tool_count = 0

        def publish(fragment):
            nonlocal tool_count
            for delta in parser.feed(fragment):
                tool_count += len(delta.get('tool_calls', []))
                if not request.parallel_tools and tool_count > 1:
                    raise ValueError('model produced multiple calls with parallel_tool_calls=false')
                emit(delta)

        reason = 'length'
        sampled = 0
        for index in range(request.max_tokens):
            check()
            token = self.sample(request, rng)
            sampled += 1
            if token in self.codec.eos_ids:
                reason = 'stop'
                break
            generated.append(token)
            if hasattr(self.codec, 'token_bytes'):
                decoded += decoder.decode(self.codec.token_bytes(token), final=False)
            else:
                # Injected codecs may expose only whole-sequence decoding.
                decoded = self.codec.decode(generated).rstrip('\ufffd')
            cut = min((decoded.find(stop) for stop in request.stop if stop in decoded), default=-1)
            if cut >= 0:
                publish(decoded[len(visible):cut])
                visible = decoded[:cut]
                reason = 'stop'
                break
            hold = max((size for stop in request.stop for size in range(1, len(stop))
                        if decoded.endswith(stop[:size])), default=0)
            safe = decoded[:-hold] if hold else decoded
            publish(safe[len(visible):])
            visible = safe
            if index + 1 < request.max_tokens:
                check()
                self.forward([token], logits=request.temperature > 0)
        else:
            reason = 'length'
        if reason != 'stop' or not any(stop in decoded for stop in request.stop):
            final_decoded = (decoded + decoder.decode(b'', final=True)
                             if hasattr(self.codec, 'token_bytes') else self.codec.decode(generated))
            publish(final_decoded[len(visible):])
            visible = final_decoded
        truncated = reason == 'length'
        for delta in parser.finish(truncated=truncated):
            tool_count += len(delta.get('tool_calls', []))
            if not request.parallel_tools and tool_count > 1:
                raise ValueError('model produced multiple calls with parallel_tool_calls=false')
            emit(delta)
        message = parse_assistant(('<think>' if request.thinking else '') + visible,
                                  tools=request.tools, call_id_prefix=parser.call_id_prefix,
                                  truncated=truncated)
        calls = message.get('tool_calls', [])
        if not request.parallel_tools and len(calls) > 1:
            raise ValueError('model produced multiple calls with parallel_tool_calls=false')
        if calls and not truncated:
            reason = 'tool_calls'
        usage = {'prompt_tokens': len(request.prompt), 'completion_tokens': sampled,
                 'total_tokens': len(request.prompt) + sampled}
        return message, reason, usage


class Server(ThreadingHTTPServer):
    """Bounded HTTP workers, with one non-queued inference slot."""
    daemon_threads = True
    request_queue_size = 16
    allow_reuse_address = True

    def __init__(self, address, engine):
        self.engine = engine
        self.workers = threading.BoundedSemaphore(16)
        super().__init__(address, Handler)

    def process_request(self, request, address):
        if not self.workers.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.workers.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.workers.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'Redshift/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *_):
        # Never log request paths, bodies, prompts or generated tool arguments.
        pass

    def send_json(self, status, body):
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(encoded)

    def do_GET(self):
        engine = self.server.engine
        try:
            if self.path == '/health':
                self.send_json(200 if engine.ready else 503,
                               {'ready': engine.ready, 'busy': engine.busy, 'model': engine.model_id,
                                'context': engine.context, 'engine': 'redshift'})
            elif self.path == '/v1/models':
                self.send_json(200, {'object': 'list', 'data': [
                    {'id': engine.model_id, 'object': 'model', 'created': engine.created,
                     'owned_by': 'redshift', 'context_length': engine.context}]})
            else:
                self.send_json(404, APIError('unknown endpoint', 404).payload())
        except OSError:
            pass

    def read_body(self):
        if self.headers.get('Transfer-Encoding'):
            raise APIError('Transfer-Encoding is not supported')
        lengths = self.headers.get_all('Content-Length', [])
        if len(lengths) != 1:
            raise APIError('one Content-Length header is required', 411)
        try:
            size = int(lengths[0])
        except ValueError as error:
            raise APIError('invalid Content-Length') from error
        if size > MAX_BODY:
            raise APIError(f'request body exceeds {MAX_BODY} bytes', 413)
        if size < 1:
            raise APIError('request body must not be empty')
        content_type = self.headers.get('Content-Type', '').split(';')[0].strip().lower()
        if content_type != 'application/json':
            raise APIError('Content-Type must be application/json', 415)
        try:
            body = self.rfile.read(size)
            if len(body) != size:
                raise Cancelled()
            return json.loads(body, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError) as error:
            raise APIError('body must be valid UTF-8 JSON without NaN/Infinity') from error

    def disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if readable and self.connection.recv(1, socket.MSG_PEEK) == b'':
                return True
        except OSError:
            return True
        return False

    def do_POST(self):
        engine = self.server.engine
        locked, streaming = False, False
        request_id, created = 'chatcmpl-' + uuid.uuid4().hex, int(time.time())
        last_heartbeat = time.monotonic()

        def event(payload):
            self.wfile.write(('data: ' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n\n').encode())
            self.wfile.flush()

        def chunk(delta, finish=None, usage=None, choices=True):
            value = {'id': request_id, 'object': 'chat.completion.chunk', 'created': created,
                     'model': engine.model_id, 'choices':
                         [{'index': 0, 'delta': delta, 'finish_reason': finish}] if choices else []}
            if request.include_usage:
                value['usage'] = usage
            event(value)

        def check():
            nonlocal last_heartbeat
            if not engine.ready or self.disconnected():
                raise Cancelled()
            if streaming and time.monotonic() - last_heartbeat >= 5:
                self.wfile.write(b': keepalive\n\n')
                self.wfile.flush()
                last_heartbeat = time.monotonic()

        try:
            if self.path != '/v1/chat/completions':
                raise APIError('unknown endpoint', 404)
            data = self.read_body()
            if not engine.ready:
                raise APIError('engine is unavailable', 503, 'engine_unavailable')
            if not engine.lock.acquire(blocking=False):
                raise APIError('engine is busy; retry after the active completion finishes', 429, 'engine_busy')
            locked = True
            request = engine.prepare(data)
            if request.stream:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                streaming = True
                chunk({'role': 'assistant'})
            message, finish, usage = engine.generate(request, check, chunk if streaming else lambda _: None)
            if streaming:
                chunk({}, finish)
                if request.include_usage:
                    chunk({}, usage=usage, choices=False)
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            else:
                self.send_json(200, {'id': request_id, 'object': 'chat.completion', 'created': created,
                                    'model': engine.model_id, 'choices': [
                                        {'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage})
        except (Cancelled, BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as error:
            if not isinstance(error, APIError):
                if locked:
                    engine.recover()
                LOG.error('Completion failed (%s)', type(error).__name__)
                error = APIError('inference failed; see server diagnostics', 500, 'inference_error')
            try:
                if streaming:
                    event(error.payload())
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                else:
                    self.send_json(error.status, error.payload())
            except OSError:
                pass
        finally:
            if locked:
                engine.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='path to the supported Qwen GGUF')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--context', type=int, default=139264)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.context <= 139264:
        parser.error('port must be 1..65535 and context 1..139264')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from .runtime import Runtime
    from .text import TextCodec
    runtime = None
    server = None
    try:
        runtime = Runtime(args.model, context=args.context)
        codec = TextCodec(runtime.model.metadata)
        server = Server((args.host, args.port), Engine(runtime, codec, context=args.context))
        LOG.info('Ready: %s on %s:%d, context %d', MODEL_ID, args.host, args.port, args.context)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'error: {error}\n')
    finally:
        if server is not None:
            server.engine.ready = False
            server.server_close()
        if runtime is not None:
            if server is None:
                runtime.close()
            else:
                # Daemon HTTP workers must leave native inference before destruction.
                with server.engine.lock:
                    runtime.close()


if __name__ == '__main__':
    main()
