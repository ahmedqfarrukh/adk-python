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

from unittest.mock import MagicMock

import pytest

pytest.importorskip("pydantic_monty")

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors._monty_code_executor import MontyCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.genai import types


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Provides a mock InvocationContext."""
  mock_agent = MagicMock(spec=BaseAgent)
  mock_session = MagicMock(spec=Session)
  mock_session_service = MagicMock(spec=BaseSessionService)
  return InvocationContext(
      invocation_id="test_invocation",
      agent=mock_agent,
      session=mock_session,
      session_service=mock_session_service,
  )


class TestMontyCodeExecutor:

  def test_init_is_stateful_by_default(self):
    """A freshly created MontyCodeExecutor is stateful."""
    executor = MontyCodeExecutor()

    assert executor.stateful is True

  def test_execute_simple_code_returns_stdout(
      self, mock_invocation_context: InvocationContext
  ):
    """Printing in the sandbox is captured as stdout with empty stderr."""
    executor = MontyCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == "hello\n"
    assert result.stderr == ""

  def test_execute_returns_final_expression_value(
      self, mock_invocation_context: InvocationContext
  ):
    """The value of the final expression is appended to stdout."""
    executor = MontyCodeExecutor()
    code_input = CodeExecutionInput(code="1 + 2")

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "3"
    assert result.stderr == ""

  def test_sync_external_function_is_dispatched(
      self, mock_invocation_context: InvocationContext
  ):
    """A synchronous external function's return value is usable in code."""

    def search(query: str) -> str:
      """Searches for the query."""
      return f"result for {query}"

    executor = MontyCodeExecutor(external_functions={"search": search})
    code_input = CodeExecutionInput(code='print(search("kittens"))')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "result for kittens\n"
    assert result.stderr == ""

  def test_async_external_function_is_awaited(
      self, mock_invocation_context: InvocationContext
  ):
    """An async external function is awaited via the worker-thread bridge."""

    async def fetch(url: str) -> str:
      """Fetches the url."""
      return f"fetched {url}"

    executor = MontyCodeExecutor(external_functions={"fetch": fetch})
    code_input = CodeExecutionInput(code='print(await fetch("http://x"))')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "fetched http://x\n"
    assert result.stderr == ""

  def test_state_persists_across_calls_with_same_execution_id(
      self, mock_invocation_context: InvocationContext
  ):
    """Variables defined in one block are visible in a later block.

    Setup: one executor, two execute_code calls sharing execution_id "s1".
    Act: first block defines `x`, second block reads it.
    Assert: the second block sees the persisted value.
    """
    executor = MontyCodeExecutor()

    executor.execute_code(
        mock_invocation_context,
        CodeExecutionInput(code="x = 99", execution_id="s1"),
    )
    result = executor.execute_code(
        mock_invocation_context,
        CodeExecutionInput(code="print(x)", execution_id="s1"),
    )

    assert result.stdout == "99\n"
    assert result.stderr == ""

  def test_runtime_error_is_surfaced_as_stderr(
      self, mock_invocation_context: InvocationContext
  ):
    """Code that raises maps to non-empty stderr, keeping prior prints."""
    executor = MontyCodeExecutor()
    code_input = CodeExecutionInput(
        code='print("before")\nraise ValueError("boom")'
    )

    result = executor.execute_code(mock_invocation_context, code_input)

    assert "before" in result.stdout
    assert result.stderr != ""
    assert "boom" in result.stderr

  def test_code_content_returns_content_with_single_text_part(self):
    """code_content returns a user-role Content with exactly one text Part."""
    executor = MontyCodeExecutor()

    content = executor.code_content()

    assert isinstance(content, types.Content)
    assert content.role == "user"
    assert len(content.parts) == 1
    assert content.parts[0].text

  def test_code_content_documents_functions_and_limits(self):
    """Description includes each function's signature, docstring, and limits."""

    def lookup(item_id: int) -> str:
      """Looks up an item by id."""
      return str(item_id)

    executor = MontyCodeExecutor(external_functions={"lookup": lookup})

    instructions = executor.code_content().parts[0].text

    assert "def lookup(item_id: int) -> str:" in instructions
    assert "Looks up an item by id." in instructions
    assert "No class definitions" in instructions
    assert "No `match` statements" in instructions
    assert "No third-party libraries" in instructions

  def test_code_content_states_no_os_access_when_callback_unset(self):
    """Without an os_callback, the description states there is no host access."""
    executor = MontyCodeExecutor()

    instructions = executor.code_content().parts[0].text

    assert (
        "no filesystem, environment, network, or clock access" in instructions
    )

  def test_code_content_uses_os_description_when_callback_present(self):
    """With an os_callback + os_description, the description is surfaced."""

    def os_callback(function_name, args, kwargs):
      return None

    executor = MontyCodeExecutor(
        os_callback=os_callback,
        os_description="You may read env vars via `os.getenv('NAME')`.",
    )

    instructions = executor.code_content().parts[0].text

    assert "You may read env vars via `os.getenv('NAME')`." in instructions
    # The generic no-access fallback must not be used when a callback is set.
    assert "no filesystem, environment, network, or clock access" not in (
        instructions
    )

  def test_code_content_uses_generic_os_note_without_description(self):
    """An os_callback with no description falls back to a generic OS note."""

    def os_callback(function_name, args, kwargs):
      return None

    executor = MontyCodeExecutor(os_callback=os_callback)

    instructions = executor.code_content().parts[0].text

    assert "A controlled host (OS) surface is available" in instructions
    assert "no filesystem, environment, network, or clock access" not in (
        instructions
    )

  def test_os_callback_getenv_is_dispatched_in_sandbox(
      self, mock_invocation_context: InvocationContext
  ):
    """Sandboxed code reads env vars routed through a raw os_callback."""

    def os_callback(function_name, args, kwargs):
      if function_name == "os.getenv" and args and args[0] == "FOO":
        return "bar"
      return None

    executor = MontyCodeExecutor(os_callback=os_callback)
    code_input = CodeExecutionInput(code='import os\nprint(os.getenv("FOO"))')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "bar\n"
    assert result.stderr == ""

  def test_os_callback_returns_not_handled_falls_back(
      self, mock_invocation_context: InvocationContext
  ):
    """Returning NOT_HANDLED defers to Monty's default unhandled behavior."""
    from pydantic_monty import NOT_HANDLED

    def os_callback(function_name, args, kwargs):
      return NOT_HANDLED

    executor = MontyCodeExecutor(os_callback=os_callback)
    code_input = CodeExecutionInput(code='import os\nprint(os.getenv("FOO"))')

    result = executor.execute_code(mock_invocation_context, code_input)

    # Monty's default unhandled OS behavior raises, surfaced as stderr.
    assert result.stderr != ""

  def test_os_callback_works_with_async_external_function(
      self, mock_invocation_context: InvocationContext
  ):
    """The async run path also dispatches OS calls through the raw callback."""

    def os_callback(function_name, args, kwargs):
      if function_name == "os.getenv" and args and args[0] == "FOO":
        return "bar"
      return None

    async def fetch(url: str) -> str:
      """Fetches the url."""
      return f"fetched {url}"

    executor = MontyCodeExecutor(
        external_functions={"fetch": fetch}, os_callback=os_callback
    )
    code_input = CodeExecutionInput(
        code=(
            'import os\nprint(os.getenv("FOO"))\nprint(await fetch("http://x"))'
        )
    )

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "bar\nfetched http://x\n"
    assert result.stderr == ""

  def test_code_content_text_is_printable(self, capsys):
    """The full description renders (functions + OS) for human review."""

    def search(query: str) -> str:
      """Searches for the query."""
      return query

    def os_callback(function_name, args, kwargs):
      return None

    executor = MontyCodeExecutor(
        external_functions={"search": search},
        os_callback=os_callback,
        os_description="Env vars are readable via `os.getenv('NAME')`.",
    )

    instructions = executor.code_content().parts[0].text
    print(instructions)

    captured = capsys.readouterr()
    assert "## Host (OS) access" in captured.out
    assert "## Available functions" in captured.out
    assert "def search(query: str) -> str:" in captured.out
    assert "os.getenv(" in captured.out

  def test_evicted_session_loses_its_persisted_state(
      self, mock_invocation_context: InvocationContext
  ):
    """An LRU-evicted session rebuilds a fresh REPL, forfeiting its variables.

    Setup: max_repl_sessions=1, define `x` under execution_id "a".
    Act: run a different execution_id "b" (evicting "a"), then re-read `x`
      under "a".
    Assert: "a"'s variable is gone, so the read errors into stderr.
    """
    executor = MontyCodeExecutor(max_repl_sessions=1)

    executor.execute_code(
        mock_invocation_context,
        CodeExecutionInput(code="x = 1", execution_id="a"),
    )
    executor.execute_code(
        mock_invocation_context,
        CodeExecutionInput(code="y = 2", execution_id="b"),
    )
    result = executor.execute_code(
        mock_invocation_context,
        CodeExecutionInput(code="print(x)", execution_id="a"),
    )

    assert result.stderr != ""
    assert executor._get_or_create_repl.cache_info().currsize == 1

  def test_non_stateful_execution_uses_throwaway_repl(
      self, mock_invocation_context: InvocationContext
  ):
    """With no execution_id, state does not persist across calls."""
    executor = MontyCodeExecutor()

    executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="z = 5")
    )
    result = executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="print(z)")
    )

    assert result.stderr != ""
