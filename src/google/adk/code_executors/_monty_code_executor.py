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
from typing import Awaitable
from typing import Callable
from typing import Protocol

from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from pydantic_monty import CollectString
from pydantic_monty import MontyError
from pydantic_monty import MontyRepl
from pydantic_monty import MontyRuntimeError
from pydantic_monty.os_access import OsFunction
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)

# Type of the callback Monty's `os=` run parameter accepts: it receives the
# operation name plus its positional/keyword args and returns the result, or
# `pydantic_monty.NOT_HANDLED` to fall back to Monty's default behavior.
OsCallback = Callable[[OsFunction, tuple[Any, ...], dict[str, Any]], Any]


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


def _run_async_blocking(awaitable: Awaitable[Any]) -> Any:
  """Runs `awaitable` to completion on a separate thread with a fresh loop.

  Used only when an external function is a coroutine function, so the
  synchronous `execute_code` can drive Monty's async `feed_run_async`. Running
  on a dedicated thread avoids `asyncio.run` failing when the calling thread
  already has a running event loop (the ADK flow's loop).

  `feed_run_async` returns a custom awaitable (not a native coroutine), so it is
  wrapped in a coroutine before being handed to `asyncio.run` -- on Python 3.11
  `asyncio.run` rejects non-coroutine awaitables outright.

  Args:
    awaitable: The awaitable to run to completion.

  Returns:
    The value produced by the awaitable.
  """
  result: dict[str, Any] = {}

  async def _drive() -> Any:
    return await awaitable

  def _runner() -> None:
    try:
      result['value'] = asyncio.run(_drive())
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

  User-declared external functions and an optional OS callback are exposed to
  the executed code, mirroring Monty's own ``external_functions`` / ``os``
  parameters. These are independent of any framework tools; redirecting an
  external function to a framework tool is left to the caller. State persists
  across code blocks via a per-execution-id MontyRepl.

  NOTE: A single MontyCodeExecutor instance carries one set of external
  functions / OS callback and one REPL cache, so it must NOT be used by
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

  os_callback: OsCallback | None = Field(default=None, exclude=True)
  """Optional callback for Monty's ``os=`` run surface (filesystem/env/clock).

  Passed straight to ``feed_run`` / ``feed_run_async``. It is invoked with
  ``(function_name, args, kwargs)`` -- e.g. ``('os.getenv', ('FOO',), {})`` --
  and must return the operation's result, or ``pydantic_monty.NOT_HANDLED`` to
  fall back to Monty's default unhandled behavior. None means no host access:
  every OS call falls through to that default. Excluded from serialization."""

  os_description: str | None = Field(default=None, exclude=True)
  """Optional Markdown describing what ``os_callback`` supports, surfaced to the
  model via ``code_instructions``. The callback is opaque (a plain function), so
  there is nothing to introspect -- supply this to keep the prompt and the
  callback's real behavior aligned. Ignored when ``os_callback`` is None."""

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
        #
        # The `feed_run_async` stub types `os` as `AbstractOS`, but Monty only
        # ever calls it as `os(function_name, args, kwargs)` -- exactly the
        # `os_callback` contract. A plain callable is verified to work here at
        # runtime (the sync `feed_run` path is typed for a raw callable), so we
        # pass it directly and silence the stub-only mismatch.
        output = _run_async_blocking(
            repl.feed_run_async(
                code_execution_input.code,
                external_functions=self.external_functions,
                print_callback=collector,
                os=self.os_callback,  # type: ignore[arg-type]
            )
        )
      else:
        # No async functions: run directly, no event loop, no extra thread.
        output = repl.feed_run(
            code_execution_input.code,
            external_functions=self.external_functions,
            print_callback=collector,
            os=self.os_callback,
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
    OS access) is fixed for the lifetime of the executor instance and does not
    vary per turn. It is derived from the same ``external_functions`` and
    ``os_callback`` / ``os_description`` that drive execution, so the prompt and
    the type-check stubs stay aligned.

    Returns:
      A Markdown snippet documenting the sandbox limits, the host (OS) access,
      the external functions (with typed signatures and docstrings), and the
      I/O contract.
    """
    sections = [_MONTY_LANGUAGE_LIMITS]

    sections.append('## Host (OS) access')
    if self.os_callback is not None:
      # The callback is opaque, so we cannot enumerate operations; rely on the
      # caller-supplied description (or a generic note when none is given).
      sections.append(
          self.os_description
          or (
              'A controlled host (OS) surface is available through the usual'
              ' `pathlib.Path`, `os`, and `datetime` calls. Network access is'
              ' not available.'
          )
      )
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
