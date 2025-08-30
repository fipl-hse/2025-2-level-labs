.. _run-in-terminal-label:

Run Python Programs in Terminal
===============================

This is a short tutorial on calling Python programs in terminal (Unix)
or ``cmd`` (Windows).

Run simple program with no dependencies
---------------------------------------

Let’s say you have a Python program represented as ``script.py`` file.

.. code:: python

   def main():
       print("Hello world!")

   if __name__ == "__main__":
       main()

To run your program, you need to open terminal (Windows: ``Powershell``,
macOS: ``Terminal``).

Then activate environment:

1. For Windows: ``.\venv\Scripts\activate``.
2. For macOS: ``source venv\bin\activate``.

If you have problems with access in ``PowerShell``, you should change
the execution policy in 2 steps:

1. Start Windows ``PowerShell`` with the “Run as Administrator” option.
2. Enable running unsigned scripts by entering:

.. code:: text

   set-executionpolicy remotesigned

If any other problems appear during activation, write in chat to get
help.

As a result, you get ``(venv)`` at the beginning of line.

Now you run your program ``python script.py``
and get output ``Hello world!``.

The program executes fine if you call Python from the same directory
where script resides. To go to the script directory use
``cd "path_to_directory"`` command. Or call python using the full path
to the script e.g.:

.. code:: shell

   python "full_path_to_directory_with_script/script.py"

If ``python`` is not recognized as a command, you need to install it.
Follow install instructions from :ref:`starting-guide-label`.

Running programs with custom modules
------------------------------------

Now, try to run your scraper:

1. Go to your project’s folder
   ``cd C:\Users\user\Documents\2025-2-level-ctlr-admin``.
2. Run scraper ``python lab_5_scraper/scraper.py``.

You get error:

.. code:: shell

   Traceback (most recent call):
      File ...., line 5 in <module>
         from core_utils.constants import ASSETS_PATH
   ModuleNotFoundError: No module named 'core_utils'

Why? When we run the same from our PyCharm, it works like a charm! What
is wrong with terminal run? ``core_utils`` is exactly here, in our
current directory!

Answer is a bit deep. We need to understand how Python imports modules
and libraries. To be able to import installed libraries, Python needs to
know where they are placed. By default, it knows several locations on
your computer where it can find requested library.

You can see them if you want ``python -c "import sys;print(sys.path)"``.
You will see standard directories where Python will try to find
libraries:

.. code:: text

   [
       '/Users/alexanderdemidovskij/.pyenv/versions/3.10.8/lib/python310.zip',
       '/Users/alexanderdemidovskij/.pyenv/versions/3.10.8/lib/python3.10',
       '/Users/alexanderdemidovskij/.pyenv/versions/3.10.8/lib/python3.10/lib-dynload',
       '/Users/alexanderdemidovskij/Documents/hse/2022-2-level-ctlr/venv/lib/python3.10/site-packages'
   ]

So, if the module you are trying to import is not within these paths,
Python will fail with the aforementioned error.

To resolve this you need to explicitly add path to your custom
dependencies to that standard list.

The recommended way is to append it to a ``PYTHONPATH`` system variable
in terminal:

1. For Windows: ``$env:PYTHONPATH = "$pwd;" + $env:PYTHONPATH``.
2. For macOS: ``export PYTHONPATH=$pwd:$PYTHONPATH``, ``pwd`` allows to
   get current working directory.

Run scraper again ``python lab_5_scraper/scraper.py``.

Does it work? If yes, congratulations, you have a chance to get the
highest possible mark. If not, write in chat for help.
