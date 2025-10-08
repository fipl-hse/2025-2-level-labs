.. _running-tests-label:

Working with tests: locally and in CI
=====================================

Running tests locally with Visual Studio Code
---------------------------------------------

To configure tests locally you need to perform several steps:

1. Install tests dependencies:

   .. code:: bash

      python -m pip install -r requirements_qa.txt

.. important:: Ensure you have activated your environment
               if you have such by running ``.\venv\Scripts\activate``
               (Windows) or ``source venv\bin\activate`` (macOS).

2. Create a new configuration:

   To create a new configuration open the Testing tab on the side
   bar of Visual Studio Code and press the `Configure Python Tests`
   button.

   .. image:: ../images/tests/vscode_testing_tab.jpg

   Alternatively, you can open configuration settings via command bar.
   Use `Ctrl + Shift + P` keyboard shortcut to open it and type in
   `Python: Configure Tests`.

   .. image:: ../images/tests/vscode_command_bar.jpg

3. Choose ``pytest`` as a target:

   .. image:: ../images/tests/vscode_tests_configuration_step_1.jpg

4. Choose the directory to run all tests. You can use root directory to run all
   tests or a specific lab.

   .. image:: ../images/tests/vscode_tests_configuration_step_2.jpg

   When you are done, the `settings.json` file for the tests will be opened
   and all the tests will be displayed on the `Testing` tab of the
   Visual Studio Code.

   .. image:: ../images/tests/vscode_configured_tests.jpg

   To run the test, press the run button, as indicated in the screenshot above.

6. As you have some tests failing, you want to run them separately. You can press
   a run button next to a test you want to run in the tests files specifically
   or in the `Testing` tab.

   .. image:: ../images/tests/vscode_running_tests.jpg


7. When you want to debug a test, execute debugging by clicking a run button
   with a bug on it on a test you want to run in the `Testing` tab or make a
   right click on the testing button in the test file itself and choose the
   `Debug Test` option.

   .. image:: ../images/tests/vscode_debugging.jpg

   To debug you should put a breakpoint in your code or in the test itself.
   Breakpoints are red dots that you can put at the potentially vulnerable place of code.
   The execution stops at breakpoints and you can debug your code from these lines.

   .. image:: ../images/tests/breakpoints.jpg


Running tests in command-line
-----------------------------

1. Install dependencies (assuming you have activated the environment
   from the previous step):

   .. code:: bash

      python -m pip install -r requirements_qa.txt

2. Run the tests for the given mark. You can select any level:
   ``mark4``, ``mark6``, ``mark8``, ``mark10``:

   .. code:: bash

      python -m pytest -m mark8

   To run tests for a specific laboratory work you can add the directory name
   after `pytest` command. The full terminal output should look like that:

   .. image:: ../images/tests/running_from_command_line.jpg

   .. hint:: Note that if you activated virtual environment and installed
            requirements properly, you can use `pytest` without calling
            `python -m` first.

Running tests in CI
-------------------

Tests will never run until you create a Pull Request.

The very first check happens exactly when you create a pull request.
After that, each time you push changes in your fork, CI check will be
automatically started, normally within a minute or two. To see the
results, navigate to your PR and click either the particular step in the
report at the end of a page, or click **Checks** in the toolbar.

.. image:: ../images/tests/ci_report.png

.. image:: ../images/tests/ci_tab.png

Inspect each step by clicking through the list to the left.
