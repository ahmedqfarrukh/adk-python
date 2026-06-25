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

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult


class _MinimalCodeExecutor(BaseCodeExecutor):
  """A minimal executor that only implements the required abstract method."""

  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    return CodeExecutionResult(stdout="", stderr="", output_files=[])


class TestBaseCodeExecutor:

  def test_code_content_defaults_to_none(self):
    """An executor that does not override code_content contributes nothing."""
    executor = _MinimalCodeExecutor()

    assert executor.code_content() is None
