# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import get_args
from typing import Protocol

from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from pydantic_monty import CollectString
from pydantic_monty import MontyError
from pydantic_monty import MontyRepl
from pydantic_monty import MontyRuntimeError
from pydantic_monty.os_access import AbstractOS
from pydantic_monty.os_access import OsFunction
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)


# Maps each Monty `OsFunction` literal to the `AbstractOS` method that backs it.
# This mirrors `AbstractOS.dispatch`, so the prompt instructions, the runtime
# dispatch, and the type checker never drift. The `OsFunction` Literal is the
# source of truth for the set of operations (see `_handled_os_functions`).
_OS_FUNCTION_TO_METHOD: dict[OsFunction, str] = {
    'Path.exists': 'path_exists',
    'Path.is_file': 'path_is_file',
    'Path.is_dir': 'path_is_dir',
    'Path.is_symlink': 'path_is_symlink',
    'Open': 'path_open',
    'Path.read_text': 'path_read_text',
    'Path.read_bytes': 'path_read_bytes',
    'Path.write_text': 'path_write_text',
    'Path.write_bytes': 'path_write_bytes',
    'Path.append_text': 'path_append_text',
    'Path.append_bytes': 'path_append_bytes',
    'Path.mkdir': 'path_mkdir',
    'Path.unlink': 'path_unlink',
    'Path.rmdir': 'path_rmdir',
    'Path.iterdir': 'path_iterdir',
    'Path.stat': 'path_stat',
    'Path.rename': 'path_rename',
    'Path.resolve': 'path_resolve',
    'Path.absolute': 'path_absolute',
    'os.getenv': 'getenv',
    'os.environ': 'get_environ',
    'date.today': 'date_today',
    'datetime.now': 'datetime_now',
}

# LLM-facing call syntax for each `OsFunction`. Rendered into the instructions so
# the model knows exactly how to invoke an available host operation.
_OS_FUNCTION_USAGE: dict[OsFunction, str] = {
    'Path.exists': "Path('p').exists() -> bool",
    'Path.is_file': "Path('p').is_file() -> bool",
    'Path.is_dir': "Path('p').is_dir() -> bool",
    'Path.is_symlink': "Path('p').is_symlink() -> bool",
    'Open': "open('p', mode) -> file object",
    'Path.read_text': "Path('p').read_text() -> str",
    'Path.read_bytes': "Path('p').read_bytes() -> bytes",
    'Path.write_text': "Path('p').write_text(data: str) -> int",
    'Path.write_bytes': "Path('p').write_bytes(data: bytes) -> int",
    'Path.append_text': "append text to Path('p')",
    'Path.append_bytes': "append bytes to Path('p')",
    'Path.mkdir': "Path('p').mkdir(parents=False, exist_ok=False)",
    'Path.unlink': "Path('p').unlink()",
    'Path.rmdir': "Path('p').rmdir()",
    'Path.iterdir': "Path('p').iterdir() -> Iterable[Path]",
    'Path.stat': "Path('p').stat()",
    'Path.rename': "Path('p').rename(target)",
    'Path.resolve': "Path('p').resolve() -> Path",
    'Path.absolute': "Path('p').absolute() -> Path",
    'os.getenv': "os.getenv('NAME', default=None) -> str | None",
    'os.environ': 'os.environ -> mapping of environment variables',
    'date.today': 'date.today() -> date',
    'datetime.now': 'datetime.now(tz=None) -> datetime',
}


class _ReplCacheFn(Protocol):
  """The callable returned by `functools.lru_cache`.

  Typed explicitly so call sites (and tests) can use both the call form and the
  ``cache_info()`` / ``cache_clear()`` members without tripping the type checker
  (a plain ``Callable`` exposes neither).
  """

  def __call__(self, execution_id: str) -> MontyRepl:
    ...

  def cache_info(self) -> Any:
    ...

  def cache_clear(self) -> None:
    ...


def _run_async_blocking(coro: Coroutine[Any, Any, Any]) -> Any:
  """Runs `coro` to completion on a separate thread with a fresh event loop.

  Used only when an external function is a coroutine function, so the
  synchronous `execute_code` can drive Monty's async `feed_run_async`. Running
  on a dedicated thread avoids `asyncio.run` failing when the calling thread
  already has a running event loop (the ADK flow's loop).

  Args:
    coro: The coroutine to run to completion.

  Returns:
    The value returned by the coroutine.
  """
  result: dict[str, Any] = {}

  def _runner() -> None:
    try:
      result['value'] = asyncio.run(coro)
    except BaseException as exc:  # surface inside the calling thread
      result['error'] = exc

  thread = threading.Thread(target=_runner, daemon=True)
  thread.start()
  thread.join()
  if 'error' in result:
    raise result['error']
  return result['value']


def _format_error(exc: MontyError) -> str:
  """Formats a Monty error into a readable diagnostic string.

  Prefers the structured `display(...)` output (typing/runtime/syntax errors)
  and falls back to `str(exc)` when display is unavailable.

  Args:
    exc: The Monty error to format.

  Returns:
    A human-readable diagnostic for the error.
  """
  display = getattr(exc, 'display', None)
  if callable(display):
    if isinstance(exc, MontyRuntimeError):
      preferred = ('traceback', 'full')
    else:
      preferred = ('full', 'traceback')
    for fmt in preferred:
      try:
        # `display` comes from `getattr` (typed Any); coerce to str so the
        # declared `-> str` return type holds.
        return str(display(format=fmt))
      except (TypeError, ValueError):
        continue
      except Exception:  # pylint: disable=broad-except
        break
  return str(exc)


def _annotation_str(annotation: Any) -> str:
  """Renders a parameter/return annotation for a stub or preamble signature.

  Falls back to `typing.Any` when no annotation is present.

  Args:
    annotation: The raw annotation from `inspect.Signature`.

  Returns:
    A string suitable for embedding in a Python signature.
  """
  if annotation is inspect.Signature.empty:
    return 'Any'
  if isinstance(annotation, str):
    return annotation
  return getattr(annotation, '__name__', None) or str(annotation)


def _render_signature(name: str, fn: Callable[..., Any]) -> str:
  """Renders a `def name(...) -> ret` signature line for `fn`.

  Missing annotations fall back to `Any`. Coroutine functions are rendered with
  a leading `async`, so the model knows to `await` them and Monty's type checker
  treats the call as awaitable.

  Args:
    name: The in-sandbox name the function is exposed under.
    fn: The host callable to introspect.

  Returns:
    A single-line function signature ending in `:`.
  """
  prefix = 'async def' if inspect.iscoroutinefunction(fn) else 'def'
  try:
    signature = inspect.signature(fn)
  except (TypeError, ValueError):
    return f'{prefix} {name}(*args: Any, **kwargs: Any) -> Any:'

  rendered_params = []
  for param in signature.parameters.values():
    rendered = param.name
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
      rendered = '*' + rendered
    elif param.kind == inspect.Parameter.VAR_KEYWORD:
      rendered = '**' + rendered
    if param.annotation is not inspect.Signature.empty:
      rendered += f': {_annotation_str(param.annotation)}'
    elif param.kind not in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ):
      rendered += ': Any'
    if param.default is not inspect.Parameter.empty:
      rendered += f' = {param.default!r}'
    rendered_params.append(rendered)

  return_annotation = _annotation_str(signature.return_annotation)
  return (
      f'{prefix} {name}({", ".join(rendered_params)}) -> {return_annotation}:'
  )


_MONTY_LANGUAGE_LIMITS = (
    'You write Python that runs inside the Monty sandbox, a minimal and secure'
    ' embedded Python interpreter. Call the functions documented below to'
    ' interact with the outside world.\n\n'
    '## Sandbox limits\n'
    '- Only a subset of Python is supported.\n'
    '- Supported standard-library modules are limited to roughly: `sys`,'
    ' `os`, `typing`, `asyncio`, `re`, `datetime`, and `json`.\n'
    '- No third-party libraries (no `pydantic`, `requests`, `numpy`, ...).\n'
    '- No class definitions (`class ...`).\n'
    '- No `match` statements.\n'
    '- Host access (filesystem, environment, network, clock) is only available'
    ' through the OS surface and external functions documented below.'
)


class MontyCodeExecutor(BaseCodeExecutor):
  """Executes model-generated Python inside the Monty sandbox.

  User-declared external functions and an optional OS handler are exposed to
  the executed code, mirroring Monty's own ``external_functions`` / ``os``
  parameters. These are independent of any framework tools; redirecting an
  external function to a framework tool is left to the caller. State persists
  across code blocks via a per-execution-id MontyRepl.

  NOTE: A single MontyCodeExecutor instance carries one set of external
  functions / OS handler and one REPL cache, so it must NOT be used by
  multiple agents simultaneously -- give each agent its own instance. Sharing
  one instance between an agent and its sub-agents is fine.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  stateful: bool = True
  """Whether the code executor is stateful. Defaults to True so variables
  persist across code blocks within a session."""

  external_functions: dict[str, Callable[..., Any]] = Field(
      default_factory=dict, exclude=True
  )
  """In-sandbox name -> user callable, passed straight to Monty. Callables may
  be synchronous or coroutine functions; they are the user's own (they may wrap
  a framework tool if they choose). Excluded from serialization."""

  os_handler: AbstractOS | None = Field(default=None, exclude=True)
  """Optional controlled OS surface (filesystem/env/clock). None means no host
  access; OS calls fall through to Monty's default unhandled behavior."""

  script_name: str = 'agent.py'
  """The script name shown in Monty tracebacks."""

  max_repl_sessions: int = 50
  """Upper bound on cached REPL sessions; least-recently-used sessions are
  evicted beyond this, bounding memory for a long-lived executor."""

  _get_or_create_repl: _ReplCacheFn | None = PrivateAttr(default=None)

  def model_post_init(self, _context: Any) -> None:
    # Per-instance cache, keyed solely on execution_id (a str). Bounded by
    # max_repl_sessions; least-recently-used sessions are evicted automatically.
    # Wired here (not via a method decorator) so `self` never enters the cache
    # key and the cache is GC'd with the instance.
    self._get_or_create_repl = functools.lru_cache(
        maxsize=self.max_repl_sessions
    )(self._build_repl)

  def _build_repl(self, execution_id: str) -> MontyRepl:
    # `execution_id` is unused in the body -- it exists only as the cache key.
    del execution_id
    return MontyRepl(
        script_name=self.script_name,
        type_check=True,
        type_check_stubs=self._build_type_stubs(),
    )

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    logger.debug('Executing code:\n```\n%s\n```', code_execution_input.code)

    execution_id = code_execution_input.execution_id
    if execution_id is None:
      # Non-stateful: build a fresh throwaway REPL, bypassing the cache.
      repl = self._build_repl('')
    else:
      # Always wired up in model_post_init; never None after construction.
      assert self._get_or_create_repl is not None
      repl = self._get_or_create_repl(execution_id)

    has_async = any(
        inspect.iscoroutinefunction(fn)
        for fn in self.external_functions.values()
    )
    collector = CollectString()
    try:
      if has_async:
        # feed_run_async must be awaited; bridge from this sync method on a
        # dedicated worker thread (we may be inside the flow's running loop).
        output = _run_async_blocking(
            repl.feed_run_async(
                code_execution_input.code,
                external_functions=self.external_functions,
                print_callback=collector,
                os=self.os_handler,
            )
        )
      else:
        # No async functions: run directly, no event loop, no extra thread.
        output = repl.feed_run(
            code_execution_input.code,
            external_functions=self.external_functions,
            print_callback=collector,
            os=self.os_handler,
        )
    except MontyError as exc:
      return CodeExecutionResult(
          stdout=collector.output,
          stderr=_format_error(exc),
          output_files=[],
      )

    stdout = collector.output
    if output is not None:
      # Surface the final expression value so the model sees it.
      if stdout and not stdout.endswith('\n'):
        stdout += '\n'
      stdout += str(output)
    return CodeExecutionResult(stdout=stdout, stderr='', output_files=[])

  def code_instructions(self) -> str:
    """Returns a prompt-ready Markdown snippet describing the sandbox.

    The caller injects this into the agent's instruction -- specifically the
    *static* instructions, since the sandbox API surface (external functions and
    OS operations) is fixed for the lifetime of the executor instance and does
    not vary per turn. It is derived from the same ``external_functions`` and
    ``os_handler`` that drive execution, so the prompt, the type-check stubs, and
    the runtime dispatch never drift.

    Returns:
      A Markdown snippet documenting the sandbox limits, the available OS
      operations, the external functions (with typed signatures and
      docstrings), and the I/O contract.
    """
    sections = [_MONTY_LANGUAGE_LIMITS]

    sections.append('## Host (OS) access')
    handled = self._handled_os_functions()
    if handled:
      os_lines = ['A controlled OS surface is available. You may call:']
      os_lines.extend(f'- `{_OS_FUNCTION_USAGE.get(fn, fn)}`' for fn in handled)
      os_lines.append('Network access is not available.')
      sections.append('\n'.join(os_lines))
    else:
      sections.append(
          'There is no filesystem, environment, network, or clock access.'
      )

    sections.append('## Available functions')
    if self.external_functions:
      function_blocks = []
      for name, fn in self.external_functions.items():
        signature = _render_signature(name, fn)
        docstring = inspect.getdoc(fn) or 'No description provided.'
        function_blocks.append(f'{signature}\n    """{docstring}"""')
      sections.append('\n\n'.join(function_blocks))
    else:
      sections.append('No external functions are available.')

    sections.append(
        '## Output\n'
        'Anything you `print(...)` and the value of the final expression in'
        ' your code are returned to you as the observation.'
    )
    return '\n\n'.join(sections)

  def _handled_os_functions(self) -> list[OsFunction]:
    """Returns the OS operations the configured ``os_handler`` implements.

    Iterates the ``OsFunction`` literal (the source of truth) and keeps each
    operation whose backing ``AbstractOS`` method is concretely implemented on
    the handler -- i.e. not left as an ``@abstractmethod`` stub. This is what
    keeps the instructions, the runtime dispatch, and Monty's own operation set
    aligned.

    NOTE: An operation a subclass disables at runtime by returning
    ``NOT_HANDLED`` (rather than leaving it abstract) cannot be detected
    statically, so it is still listed. Leave such operations unimplemented if
    you want them omitted from the instructions.

    Returns:
      The handled operations in the canonical ``OsFunction`` order, or an empty
      list when no ``os_handler`` is configured.
    """
    if self.os_handler is None:
      return []
    handler_type = type(self.os_handler)
    handled: list[OsFunction] = []
    for function_name in get_args(OsFunction):
      method_name = _OS_FUNCTION_TO_METHOD.get(function_name)
      if method_name is None:
        # Monty added an OsFunction we have not mapped yet; skip rather than
        # guess at how to call it.
        logger.warning('Unmapped Monty OsFunction: %s', function_name)
        continue
      method = getattr(handler_type, method_name, None)
      if method is None or getattr(method, '__isabstractmethod__', False):
        continue
      handled.append(function_name)
    return handled

  def _build_type_stubs(self) -> str:
    """Builds Monty `type_check_stubs` declarations for external functions.

    Emits one ``def name(...) -> ret: ...`` declaration per external function,
    rendered from `inspect.signature`. Kept internal; the LLM never sees these.

    Returns:
      A string of stub declarations for Monty's type checker.
    """
    lines = ['from typing import Any']
    for name, fn in self.external_functions.items():
      lines.append(f'{_render_signature(name, fn)} ...')
    return '\n'.join(lines)
